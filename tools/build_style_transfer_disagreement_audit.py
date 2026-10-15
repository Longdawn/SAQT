import argparse
import csv
import html
import json
import pickle
import random
from collections import Counter
from pathlib import Path

from PIL import Image


DOMAINS = ("zhanguoce_song", "tangshi_xuanxuan", "shujiao_ming")
SPLITS = ("train", "valid", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a stratified human audit from frozen recognizer disagreements."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/style_stratified_transfer_pilot_v2_candidate"),
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path(
            "/home/ubuntu/mmocr/work_dirs/style_transfer_annotation_mining_0803"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/notes/style_transfer_annotation_audit_0803"),
    )
    parser.add_argument("--risk-per-domain", type=int, default=20)
    parser.add_argument("--random-per-domain", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def levenshtein(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def similarity(left, right):
    denominator = max(len(left), len(right), 1)
    return 1.0 - levenshtein(left, right) / denominator


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(path):
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    output = {}
    for row in rows:
        sample_id = Path(row["img_path"]).stem
        output[sample_id] = {
            "gt": row["gt_text"]["item"],
            "pred": row["pred_text"]["item"],
        }
    return output


def enrich(row, crnn, svtr):
    sample_id = row["id"]
    if sample_id not in crnn or sample_id not in svtr:
        raise KeyError(f"Missing frozen prediction for {sample_id}")
    if crnn[sample_id]["gt"] != row["text"] or svtr[sample_id]["gt"] != row["text"]:
        raise ValueError(f"Prediction/manifest label mismatch for {sample_id}")

    label = row["text"]
    crnn_text = crnn[sample_id]["pred"]
    svtr_text = svtr[sample_id]["pred"]
    crnn_label_similarity = similarity(crnn_text, label)
    svtr_label_similarity = similarity(svtr_text, label)
    model_similarity = similarity(crnn_text, svtr_text)
    exact_consensus_against = crnn_text == svtr_text and crnn_text != label
    first_char_consensus_against = bool(
        label
        and crnn_text
        and svtr_text
        and crnn_text[0] == svtr_text[0]
        and crnn_text[0] != label[0]
    )
    suspicion_score = (
        model_similarity
        - (crnn_label_similarity + svtr_label_similarity) / 2
        + 2.0 * exact_consensus_against
        + 0.75 * first_char_consensus_against
    )

    if crnn_text == label and svtr_text == label:
        category = "both_match"
    elif exact_consensus_against:
        category = "models_agree_against"
    elif (crnn_text == label) != (svtr_text == label):
        category = "one_matches"
    else:
        category = "both_disagree"

    return {
        **row,
        "crnn_text": crnn_text,
        "svtr_text": svtr_text,
        "category": category,
        "crnn_label_similarity": round(crnn_label_similarity, 6),
        "svtr_label_similarity": round(svtr_label_similarity, 6),
        "model_similarity": round(model_similarity, 6),
        "exact_consensus_against": exact_consensus_against,
        "first_char_consensus_against": first_char_consensus_against,
        "suspicion_score": round(suspicion_score, 6),
    }


def select_domain(rows, risk_count, random_count, seed):
    if len(rows) < risk_count + random_count:
        raise ValueError(
            f"Requested {risk_count + random_count} audit rows from {len(rows)} candidates"
        )

    exact_consensus = sorted(
        (row for row in rows if row["exact_consensus_against"]),
        key=lambda row: (-row["suspicion_score"], row["id"]),
    )
    suspicion = exact_consensus[:risk_count]
    selected_ids = {row["id"] for row in suspicion}
    suspicion.extend(
        sorted(
            (row for row in rows if row["id"] not in selected_ids),
            key=lambda row: (
                not row["first_char_consensus_against"],
                -row["suspicion_score"],
                row["id"],
            ),
        )[: risk_count - len(suspicion)]
    )
    selected_ids = {row["id"] for row in suspicion}

    # The risk set is descriptive only. The disjoint stratified random set is
    # the sole basis for estimating ordinary annotation noise.
    random_quotas = {
        "train": random_count // 4,
        "valid": (random_count - random_count // 4) // 2,
    }
    random_quotas["test"] = random_count - sum(random_quotas.values())
    random_rows = []
    for split_index, split in enumerate(SPLITS):
        candidates = sorted(
            (
                row
                for row in rows
                if row["split"] == split and row["id"] not in selected_ids
            ),
            key=lambda row: row["id"],
        )
        random.Random(seed + split_index).shuffle(candidates)
        random_rows.extend(candidates[: random_quotas[split]])

    if len(random_rows) != random_count:
        raise ValueError(
            f"Could select only {len(random_rows)} of {random_count} random audit rows"
        )
    return [
        *({**row, "selection_type": "risk_mined"} for row in suspicion),
        *({**row, "selection_type": "stratified_random"} for row in random_rows),
    ]


def write_csv(path, rows):
    fields = [
        "review_status",
        "corrected_text",
        "review_notes",
        "id",
        "source_id",
        "title",
        "split",
        "selection_type",
        "text",
        "crnn_text",
        "svtr_text",
        "category",
        "suspicion_score",
        "crnn_label_similarity",
        "svtr_label_similarity",
        "model_similarity",
        "exact_consensus_against",
        "first_char_consensus_against",
        "page_id",
        "column_index_right_to_left",
        "thumbnail",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"review_status": "", "corrected_text": "", "review_notes": "", **row}
            )


def write_html(path, rows):
    cards = []
    for row in rows:
        cards.append(
            f"<article data-id=\"{html.escape(row['id'])}\" "
            f"data-domain=\"{html.escape(row['source_id'])}\" "
            f"data-split=\"{html.escape(row['split'])}\" "
            f"data-selection=\"{html.escape(row['selection_type'])}\">"
            f"<h2>{html.escape(row['id'])}</h2>"
            f"<p><b>{html.escape(row['source_id'])} / {row['split']} / "
            f"{html.escape(row['selection_type'])}</b></p>"
            f"<img src=\"{html.escape(row['thumbnail'])}\" alt=\"{html.escape(row['id'])}\">"
            f"<p>原标签：<strong>{html.escape(row['text'])}</strong></p>"
            f"<p>CRNN：{html.escape(row['crnn_text'])}</p>"
            f"<p>SVTR：{html.escape(row['svtr_text'])}</p>"
            f"<p>类别：{html.escape(row['category'])}；可疑度：{row['suspicion_score']}</p>"
            "<div class=\"review\"><label>审查状态<select class=\"status\">"
            "<option value=\"\">未审查</option><option value=\"correct\">标签正确</option>"
            "<option value=\"issue\">标签错误</option><option value=\"uncertain\">不确定</option>"
            "</select></label><label>修正文本<input class=\"corrected\"></label>"
            "<label>备注<input class=\"notes\"></label></div>"
            "</article>"
        )
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>跨风格标签审查</title><style>
body{font-family:sans-serif;margin:20px;background:#f4f4f2;color:#222}
header{position:sticky;top:0;z-index:2;background:#f4f4f2;border-bottom:1px solid #aaa;padding:10px 0}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}button,select,input{min-height:34px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
article{background:white;border:1px solid #bbb;padding:12px;border-radius:6px}
article.done{border-color:#287a3b}article.issue{border-color:#b42318;border-width:2px}
h2{font-size:16px;overflow-wrap:anywhere}img{display:block;max-width:100%;max-height:720px;margin:auto}
p{overflow-wrap:anywhere;line-height:1.5}strong{color:#8b1a1a}.review{display:grid;gap:7px}
.review label{display:grid;gap:3px}.hidden{display:none}</style></head><body>
<header><h1>跨风格识别标签分层审查</h1><div class="toolbar">
<label>域 <select id="domainFilter"><option value="">全部</option><option>zhanguoce_song</option><option>tangshi_xuanxuan</option><option>shujiao_ming</option></select></label>
<label>Split <select id="splitFilter"><option value="">全部</option><option>train</option><option>valid</option><option>test</option></select></label>
<label>类型 <select id="selectionFilter"><option value="">全部</option><option>risk_mined</option><option>stratified_random</option></select></label>
<label>状态 <select id="statusFilter"><option value="">全部</option><option value="pending">未审查</option><option value="correct">标签正确</option><option value="issue">标签错误</option><option value="uncertain">不确定</option></select></label>
<b id="progress"></b><button id="copyIssues">复制问题 ID</button><button id="exportJson">导出审查 JSON</button>
</div></header><main>""" + "".join(cards) + """</main><script>
const key='style-transfer-audit-v3';
const state=JSON.parse(localStorage.getItem(key)||'{}');
const cards=[...document.querySelectorAll('article')];
function save(){localStorage.setItem(key,JSON.stringify(state));update();}
function update(){
 let reviewed=0,issues=0;
 for(const card of cards){
  const id=card.dataset.id,s=state[id]||{};
  card.querySelector('.status').value=s.status||'';
  card.querySelector('.corrected').value=s.corrected_text||'';
  card.querySelector('.notes').value=s.review_notes||'';
  card.classList.toggle('done',!!s.status);card.classList.toggle('issue',s.status==='issue');
  if(s.status)reviewed++;if(s.status==='issue')issues++;
  const wantedStatus=document.querySelector('#statusFilter').value;
  const visible=(!domainFilter.value||card.dataset.domain===domainFilter.value)&&
   (!splitFilter.value||card.dataset.split===splitFilter.value)&&
   (!selectionFilter.value||card.dataset.selection===selectionFilter.value)&&
   (!wantedStatus||(wantedStatus==='pending'?!s.status:s.status===wantedStatus));
  card.classList.toggle('hidden',!visible);
 }
 progress.textContent=`已审查 ${reviewed}/${cards.length}，错误 ${issues}`;
}
for(const card of cards){
 const id=card.dataset.id;
 for(const field of ['status','corrected','notes'])card.querySelector('.'+field).addEventListener('change',e=>{
  state[id]=state[id]||{};
  const name=field==='corrected'?'corrected_text':field==='notes'?'review_notes':'status';
  state[id][name]=e.target.value;save();
 });
}
for(const id of ['domainFilter','splitFilter','selectionFilter','statusFilter'])document.querySelector('#'+id).addEventListener('change',update);
copyIssues.addEventListener('click',async()=>{
 const text=cards.filter(c=>(state[c.dataset.id]||{}).status==='issue').map(c=>c.dataset.id).join('\\n');
 await navigator.clipboard.writeText(text);copyIssues.textContent='已复制';
});
exportJson.addEventListener('click',()=>{
 const payload=cards.map(c=>({id:c.dataset.id,domain:c.dataset.domain,split:c.dataset.split,selection_type:c.dataset.selection,...(state[c.dataset.id]||{})}));
 const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='style_transfer_audit_review.json';a.click();URL.revokeObjectURL(a.href);
});
update();
</script></body></html>"""
    path.write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists: {args.output_dir}")
    thumbs = args.output_dir / "thumbs"
    thumbs.mkdir(parents=True)

    all_rows = []
    summary = {"domains": {}, "selection": {}}
    for domain_index, domain in enumerate(DOMAINS):
        manifest = load_jsonl(args.dataset_root / domain / "columns_manifest.jsonl")
        crnn = load_predictions(
            args.prediction_root
            / f"crnn_{domain}"
            / "best_recog_1-N.E.D_epoch_28.pth_predictions.pkl"
        )
        svtr = load_predictions(
            args.prediction_root
            / f"svtr_tiny_{domain}"
            / "best_recog_1-N.E.D_epoch_29.pth_predictions.pkl"
        )
        enriched = [enrich(row, crnn, svtr) for row in manifest]
        summary["domains"][domain] = {
            "samples": len(enriched),
            "categories": dict(Counter(row["category"] for row in enriched)),
        }

        selected = select_domain(
            enriched,
            args.risk_per_domain,
            args.random_per_domain,
            args.seed + 100 * domain_index,
        )
        summary["selection"][domain] = dict(
            Counter(row["selection_type"] for row in selected)
        )
        all_rows.extend(selected)

    all_rows.sort(key=lambda row: (row["source_id"], row["split"], row["selection_type"], row["id"]))
    for row in all_rows:
        image_path = args.dataset_root / row["source_id"] / "lines" / f"{row['id']}.jpg"
        thumb_path = thumbs / f"{row['id']}.jpg"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((320, 1200), Image.Resampling.LANCZOS)
            image.save(thumb_path, quality=90)
        row["thumbnail"] = f"thumbs/{thumb_path.name}"

    summary["audit_samples"] = len(all_rows)
    summary["review_policy"] = {
        "pass": "at most 1 confirmed error in each domain's 40 stratified-random rows",
        "expand": "exactly 2 random-audit errors in a domain triggers expanded review",
        "fail": "3 or more random-audit errors rejects or rebuilds that domain",
        "risk_rows": "locate concrete errors only; excluded from noise-rate estimation",
        "frozen_models_are_ground_truth": False,
    }
    write_csv(args.output_dir / "review.csv", all_rows)
    write_html(args.output_dir / "index.html", all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
