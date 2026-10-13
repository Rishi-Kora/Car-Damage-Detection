import json
from pathlib import Path

SRC = Path("dataset/extracted/CarDD_release/CarDD_COCO")
OUT = Path("yolo_data")
SPLITS = ["train", "val", "test"]


def to_yolo_line(class_id, polygon, width, height):
    points = []
    for i in range(0, len(polygon), 2):
        x = min(max(polygon[i] / width, 0), 1)
        y = min(max(polygon[i + 1] / height, 0), 1)
        points += [f"{x:.6f}", f"{y:.6f}"]
    return f"{class_id} " + " ".join(points)


def convert_split(split):
    data = json.load(open(SRC / "annotations" / f"instances_{split}2017.json"))
    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    for ann in data["annotations"]:
        labels.setdefault(ann["image_id"], []).append(ann)

    for img in data["images"]:
        w, h = img["width"], img["height"]
        lines = []
        for ann in labels.get(img["id"], []):
            for polygon in ann["segmentation"]:
                if len(polygon) >= 6:
                    lines.append(to_yolo_line(ann["category_id"] - 1, polygon, w, h))

        (lbl_dir / f"{Path(img['file_name']).stem}.txt").write_text("\n".join(lines))
        link = img_dir / img["file_name"]
        if not link.exists():
            link.symlink_to((SRC / f"{split}2017" / img["file_name"]).resolve())

    print(f"{split}: {len(data['images'])} images done")
    return data["categories"]


if __name__ == "__main__":
    for split in SPLITS:
        categories = convert_split(split)
    names = "\n".join(f"  {c['id'] - 1}: {c['name']}" for c in categories)
    (OUT / "data.yaml").write_text(
        f"path: {OUT.name}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n{names}\n")
    print("Done. Now run: python train.py")