import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

class CustomMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        super(CustomMetric, self).__init__(output_transform=output_transform, device=device)
        self.ce_loss = 0.0
        self.correct_predictions = 0
        self._num_examples = 0

    def reset(self):
        self.ce_loss = 0.0
        self.correct_predictions = 0
        self._num_examples = 0

    def update(self, output):
        # Unpack the output
        y_pred, y = output

        # CrossEntropyLoss: y_pred shape (B, 11), y shape (B, 11)
        # 8 phần tử đầu là nhãn emotion, 3 phần tử sau là emotion level
        ce_loss_emotion = F.cross_entropy(F.softmax(y_pred[:, :8], dim=1), y[:, :8])
        ce_loss_level = F.cross_entropy(F.softmax(y_pred[:, 8:], dim=1), y[:, 8:])

        # Tổng CrossEntropyLoss cho cả emotion và level
        total_ce_loss = ce_loss_emotion + ce_loss_level

        # Cộng dồn loss
        self.ce_loss += total_ce_loss.item()

        # Tính toán accuracy: kiểm tra dự đoán emotion và level
        pred_emotion = F.softmax(y_pred[:, :8], dim=1).argmax(dim=1)
        true_emotion = y[:, :8].argmax(dim=1)
        pred_level = F.softmax(y_pred[:, 8:], dim=1).argmax(dim=1)
        true_level = y[:, 8:].argmax(dim=1)

        correct_emotion = (pred_emotion == true_emotion).sum().item()
        correct_level = (pred_level == true_level).sum().item()

        # Tổng số dự đoán chính xác
        self.correct_predictions += correct_emotion + correct_level
        # Tổng số mẫu (mỗi mẫu có 2 phần: emotion và level)
        self._num_examples += 2 * y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomMetric must have at least one example before it can be computed')
        
        avg_ce_loss = self.ce_loss / self._num_examples
        accuracy = self.correct_predictions / self._num_examples
        
        return {"CrossEntropyLoss": avg_ce_loss, "Accuracy": accuracy}
