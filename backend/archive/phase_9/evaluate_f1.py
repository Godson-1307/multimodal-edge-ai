# ============================================
# PHASE 7.4 - PRECISION, RECALL AND F1
# ============================================

def calculate_metrics(ground_truth, predicted, matched):

    precision = matched / predicted if predicted > 0 else 0
    recall = matched / ground_truth if ground_truth > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


# Results from 20-sample evaluation

ground_truth_items = 35

base_predicted_items = 44
base_matched_items = 27

finetuned_predicted_items = 38
finetuned_matched_items = 28


# Calculate Base Model metrics

base_precision, base_recall, base_f1 = calculate_metrics(
    ground_truth_items,
    base_predicted_items,
    base_matched_items
)


# Calculate Fine-Tuned Model metrics

ft_precision, ft_recall, ft_f1 = calculate_metrics(
    ground_truth_items,
    finetuned_predicted_items,
    finetuned_matched_items
)


print("\n============================================")
print("PHASE 7.4 - LINE ITEM METRICS")
print("============================================")

print("\nBASE MODEL")
print("--------------------------------------------")
print("Ground Truth Items:", ground_truth_items)
print("Predicted Items:", base_predicted_items)
print("Matched Items:", base_matched_items)
print("Precision: {:.2f}%".format(base_precision * 100))
print("Recall: {:.2f}%".format(base_recall * 100))
print("F1 Score: {:.2f}%".format(base_f1 * 100))


print("\nFINE-TUNED MODEL")
print("--------------------------------------------")
print("Ground Truth Items:", ground_truth_items)
print("Predicted Items:", finetuned_predicted_items)
print("Matched Items:", finetuned_matched_items)
print("Precision: {:.2f}%".format(ft_precision * 100))
print("Recall: {:.2f}%".format(ft_recall * 100))
print("F1 Score: {:.2f}%".format(ft_f1 * 100))


print("\n============================================")
print("BASE vs FINE-TUNED")
print("============================================")

print(
    "Precision: {:.2f}% -> {:.2f}%".format(
        base_precision * 100,
        ft_precision * 100
    )
)

print(
    "Recall: {:.2f}% -> {:.2f}%".format(
        base_recall * 100,
        ft_recall * 100
    )
)

print(
    "F1 Score: {:.2f}% -> {:.2f}%".format(
        base_f1 * 100,
        ft_f1 * 100
    )
)

print("\n============================================")
print("STEP 7.4 SUCCESS")
print("============================================")