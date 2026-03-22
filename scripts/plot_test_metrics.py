# scripts/plot_test_metrics.py
import json
import matplotlib.pyplot as plt

def plot_test_metrics():
    with open("artifacts/kaggle_predictions.json", "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    # Считаем точность по классам
    class_stats = {}
    for p in predictions:
        cls = p["true_class"]
        if cls not in class_stats:
            class_stats[cls] = {"correct": 0, "total": 0}
        class_stats[cls]["total"] += 1
        if p["correct"]:
            class_stats[cls]["correct"] += 1
    
    classes = list(class_stats.keys())
    accuracies = [class_stats[c]["correct"] / class_stats[c]["total"] * 100 for c in classes]
    
    # График 1: Точность по классам
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.barh(classes, accuracies, color='skyblue')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Class')
    plt.title('Test Accuracy by Class')
    plt.xlim(0, 100)
    
    # График 2: Распределение уверенности
    confidences = [p["confidence"] for p in predictions]
    plt.subplot(1, 2, 2)
    plt.hist(confidences, bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    
    plt.tight_layout()
    plt.savefig("artifacts/test_metrics.png", dpi=150)
    plt.show()
    
    # Вывод общей точности
    total_correct = sum(1 for p in predictions if p["correct"])
    overall_acc = total_correct / len(predictions) * 100
    print(f"✅ Overall Test Accuracy: {overall_acc:.2f}%")
    print(f"✅ Графики сохранены: artifacts/test_metrics.png")

if __name__ == "__main__":
    plot_test_metrics()