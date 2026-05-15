"""
딥러닝 분류 모델 모듈 (PyTorch)
─────────────────────────────────
정형 데이터(Tabular)를 위한 Fully Connected Neural Network
BatchNorm + Dropout으로 과적합을 방지합니다.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MentalHealthNet(nn.Module):
    """
    청소년 정신건강 위험도 분류를 위한 신경망

    구조:
        입력층 → [Linear → BatchNorm → ReLU → Dropout] × 3 → 출력층

    Args:
        input_dim: 입력 피처 수 (전처리 후 컬럼 수)
        hidden_dims: 은닉층 뉴런 수 리스트 (기본: [128, 64, 32])
        output_dim: 클래스 수 (Low/Medium/High = 3)
        dropout_rate: 드롭아웃 비율 (기본 0.3)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        output_dim: int = 3,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        layers = []
        in_features = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DLClassifier:
    """
    PyTorch 신경망 학습 / 평가 / 저장을 관리하는 래퍼 클래스

    사용 예시:
        clf = DLClassifier(input_dim=12, output_dim=2, pos_weight=37.0)
        history = clf.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        clf.save("models/saved/best_dl_model.pt")
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        hidden_dims: list[int] = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 15,
        pos_weight: float = 37.0,  # 1169/31 ≈ 37 (양성 클래스 가중치)
        device: str = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.pos_weight = pos_weight  # Early Stopping 기준

        # GPU가 있으면 GPU 사용, 없으면 CPU
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[DL] 사용 디바이스: {self.device}")

        self.model: Optional[MentalHealthNet] = None
        self.history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _build_model(self) -> None:
        self.model = MentalHealthNet(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

    def _make_dataloader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    # ─────────────────────────────────────────
    # 학습
    # ─────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> dict[str, list]:
        """
        모델을 학습합니다. Early Stopping으로 과적합을 방지합니다.

        Returns:
            학습 히스토리 {"train_loss": [...], "val_loss": [...], "val_acc": [...]}
        """
        self._build_model()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # 클래스 가중치: 양성(1) 클래스를 pos_weight 배율로 강조
        weight_tensor = torch.FloatTensor([1.0, self.pos_weight]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        train_loader = self._make_dataloader(X_train, y_train, shuffle=True)
        has_val = X_val is not None and y_val is not None
        val_loader = self._make_dataloader(X_val, y_val, shuffle=False) if has_val else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        print(f"\n{'='*60}")
        print("DL 모델 학습 시작")
        print(f"  구조: {self.input_dim} → {self.hidden_dims} → {self.output_dim}")
        print(f"  에포크: {self.epochs} | 배치: {self.batch_size} | LR: {self.learning_rate}")
        print(f"  Early Stopping patience: {self.patience}")
        print(f"{'='*60}")

        for epoch in range(1, self.epochs + 1):
            # ── 학습 단계
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            self.history["train_loss"].append(train_loss)

            # ── 검증 단계
            if has_val:
                val_loss, val_acc = self._evaluate_epoch(val_loader, criterion)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                scheduler.step(val_loss)

                # Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 10 == 0 or epoch == 1:
                    print(
                        f"  Epoch {epoch:3d}/{self.epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f}"
                    )

                if patience_counter >= self.patience:
                    print(f"\n⏹ Early Stopping (epoch {epoch})")
                    break
            else:
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch:3d}/{self.epochs} | Train Loss: {train_loss:.4f}")

        # 가장 좋았던 가중치로 복원
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\n✅ 최적 모델 복원 (Best Val Loss: {best_val_loss:.4f})")

        return self.history

    def _evaluate_epoch(
        self, loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float]:
        """검증 epoch 1회 실행. (loss, accuracy) 반환."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                total_loss += loss.item()
                preds = output.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
        return total_loss / len(loader), correct / total

    # ─────────────────────────────────────────
    # 예측
    # ─────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """클래스 인덱스를 반환합니다."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            output = self.model(X_tensor)
            return output.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """각 클래스에 대한 확률을 반환합니다."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            output = self.model(X_tensor)
            probs = torch.softmax(output, dim=1)
            return probs.cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """테스트 데이터 성능 평가."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = float("nan")
        print(f"\n[DL 테스트 성능] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f}")
        return {"accuracy": acc, "f1_binary": f1, "auc_roc": auc}

    # ─────────────────────────────────────────
    # 저장 / 불러오기
    # ─────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """모델 가중치와 설정을 함께 저장합니다."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dims": self.hidden_dims,
                "dropout_rate": self.dropout_rate,
                "pos_weight": self.pos_weight,
            },
            "history": self.history,
        }, path)
        print(f"[저장] DL 모델 → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DLClassifier":
        """저장된 모델을 불러옵니다."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        obj = cls(**config)
        obj._build_model()
        obj.model.load_state_dict(checkpoint["model_state_dict"])
        obj.history = checkpoint.get("history", {})
        print(f"[로드] DL 모델 ← {path}")
        return obj
