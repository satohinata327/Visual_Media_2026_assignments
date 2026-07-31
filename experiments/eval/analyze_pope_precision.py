import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def question_id(item):
    return int(item["question_id"])


def align_generated(ground_truth, generated, method_name):
    # 公式eval_pope.pyはquestion_idの順序しか確認しないため、異なる
    # POPE設定を誤って評価しないよう、質問文と画像名も照合する。
    generated_by_id = {}
    for item in generated:
        item_id = question_id(item)
        if item_id in generated_by_id:
            raise ValueError(
                f"{method_name}: question_id={item_id}が重複しています。"
            )
        generated_by_id[item_id] = item

    aligned = []
    for gt_item in ground_truth:
        item_id = question_id(gt_item)
        if item_id not in generated_by_id:
            raise ValueError(
                f"{method_name}: question_id={item_id}の回答がありません。"
            )

        generated_item = generated_by_id[item_id]
        if (
            "prompt" in generated_item
            and generated_item["prompt"] != gt_item["text"]
        ):
            raise ValueError(
                f"{method_name}: question_id={item_id}の質問文が"
                "ground truthと一致しません。"
            )
        if (
            "image" in generated_item
            and generated_item["image"] != gt_item["image"]
        ):
            raise ValueError(
                f"{method_name}: question_id={item_id}の画像名が"
                "ground truthと一致しません。"
            )
        aligned.append(generated_item)

    if len(generated_by_id) != len(ground_truth):
        raise ValueError(
            f"{method_name}: 件数が一致しません。"
            f"ground_truth={len(ground_truth)}, "
            f"generated={len(generated_by_id)}"
        )
    return aligned


def is_correct(gt_item, generated_item):
    # 公式eval_pope.pyと同じ部分文字列判定を使用する。
    label = gt_item["label"].lower().strip()
    answer = generated_item["text"].lower().strip()
    if label == "yes":
        return "yes" in answer
    if label == "no":
        return "no" in answer
    raise ValueError(
        f"未知のground-truth label: {gt_item['label']!r}"
    )


def calculate_metrics(ground_truth, generated):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    unknown_answers = 0
    yes_answers = 0
    gate_observed = 0
    gate_triggered = 0
    gate_triggered_correct = 0
    gate_triggered_on_yes = 0
    gate_triggered_on_no = 0
    gate_triggered_on_yes_correct = 0
    gate_triggered_on_no_correct = 0
    clean_p_no_values = []

    for gt_item, generated_item in zip(ground_truth, generated):
        label = gt_item["label"].lower().strip()
        answer = generated_item["text"].lower().strip()
        correct = is_correct(gt_item, generated_item)

        if "yes" not in answer and "no" not in answer:
            unknown_answers += 1

        if label == "yes":
            if correct:
                true_positive += 1
                yes_answers += 1
            else:
                false_negative += 1
        elif label == "no":
            if correct:
                true_negative += 1
            else:
                false_positive += 1
                yes_answers += 1

        metadata = generated_item.get("metadata") or {}
        if "precision_gate_triggered" in metadata:
            gate_observed += 1
            if bool(metadata["precision_gate_triggered"]):
                gate_triggered += 1
                if correct:
                    gate_triggered_correct += 1
                if label == "yes":
                    gate_triggered_on_yes += 1
                    if correct:
                        gate_triggered_on_yes_correct += 1
                else:
                    gate_triggered_on_no += 1
                    if correct:
                        gate_triggered_on_no_correct += 1

        clean_p_no = metadata.get("clean_p_no")
        if clean_p_no is not None:
            clean_p_no_values.append(float(clean_p_no))

    total = len(ground_truth)
    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )
    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )
    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    metrics = {
        "count": total,
        "accuracy": safe_divide(
            true_positive + true_negative,
            total,
        ),
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(
            2 * precision * recall,
            precision + recall,
        ),
        "specificity_no_accuracy": specificity,
        "false_positive_rate": safe_divide(
            false_positive,
            true_negative + false_positive,
        ),
        "false_negative_rate": safe_divide(
            false_negative,
            true_positive + false_negative,
        ),
        "yes_ratio": safe_divide(yes_answers, total),
        "unknown_ratio": safe_divide(unknown_answers, total),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }

    if gate_observed:
        metrics.update(
            {
                "gate_observed_count": gate_observed,
                "gate_triggered_count": gate_triggered,
                "gate_trigger_rate": safe_divide(
                    gate_triggered,
                    gate_observed,
                ),
                "gate_triggered_accuracy": safe_divide(
                    gate_triggered_correct,
                    gate_triggered,
                ),
                "gate_triggered_on_yes_count": gate_triggered_on_yes,
                "gate_triggered_on_yes_accuracy": safe_divide(
                    gate_triggered_on_yes_correct,
                    gate_triggered_on_yes,
                ),
                "gate_triggered_on_no_count": gate_triggered_on_no,
                "gate_triggered_on_no_accuracy": safe_divide(
                    gate_triggered_on_no_correct,
                    gate_triggered_on_no,
                ),
                "mean_clean_p_no": safe_divide(
                    sum(clean_p_no_values),
                    len(clean_p_no_values),
                ),
            }
        )

    return metrics


def compare_correctness(
    ground_truth,
    before,
    after,
    before_name,
    after_name,
):
    both_correct = 0
    both_wrong = 0
    improved = 0
    degraded = 0

    for gt_item, before_item, after_item in zip(
        ground_truth,
        before,
        after,
    ):
        before_correct = is_correct(gt_item, before_item)
        after_correct = is_correct(gt_item, after_item)
        if before_correct and after_correct:
            both_correct += 1
        elif not before_correct and not after_correct:
            both_wrong += 1
        elif not before_correct and after_correct:
            improved += 1
        else:
            degraded += 1

    return {
        "before": before_name,
        "after": after_name,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "improved_wrong_to_correct": improved,
        "degraded_correct_to_wrong": degraded,
        "net_improvement": improved - degraded,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_files", type=Path, required=True)
    parser.add_argument("--regular_files", type=Path, default=None)
    parser.add_argument("--vcd_files", type=Path, default=None)
    parser.add_argument("--precision_files", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    method_paths = {
        "regular": args.regular_files,
        "vcd": args.vcd_files,
        "precision_gated_vcd": args.precision_files,
    }
    if not any(method_paths.values()):
        parser.error(
            "--regular_files、--vcd_files、--precision_filesの"
            "いずれかを指定してください。"
        )

    ground_truth = read_jsonl(args.gt_files)
    aligned_by_method = {}
    metrics_by_method = {}

    for method_name, path in method_paths.items():
        if path is None:
            continue
        generated = read_jsonl(path)
        aligned = align_generated(
            ground_truth,
            generated,
            method_name,
        )
        aligned_by_method[method_name] = aligned
        metrics_by_method[method_name] = calculate_metrics(
            ground_truth,
            aligned,
        )

    transitions = []
    comparison_pairs = [
        ("regular", "vcd"),
        ("vcd", "precision_gated_vcd"),
        ("regular", "precision_gated_vcd"),
    ]
    for before_name, after_name in comparison_pairs:
        if (
            before_name in aligned_by_method
            and after_name in aligned_by_method
        ):
            transitions.append(
                compare_correctness(
                    ground_truth,
                    aligned_by_method[before_name],
                    aligned_by_method[after_name],
                    before_name,
                    after_name,
                )
            )

    result = {
        "ground_truth": str(args.gt_files),
        "metrics": metrics_by_method,
        "transitions": transitions,
    }
    formatted = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )
    print(formatted)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            formatted + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
