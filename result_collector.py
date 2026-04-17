import os

from ecg_radar_align import batch_max_pearson_corr, align_ecg_radar
from matrices import ECGPhysioMetrics
import pandas as pd


class ResultCollector:
    def __init__(self, model_name, domain, n_fold):
        self.model_name = model_name
        self.domain = domain
        self.n_fold = n_fold

        self.ECG_physio_metrics = ECGPhysioMetrics()

        # result[epoch][fold] = dict
        # result示意
        example = {'0': {
            '0': {'pcc': 0.812, '其它指标以此类推': 733.1}, # fold 0
            '1': {'pcc': 0.833, '其它指标以此类推': 88.1}, # fold 1
            '2': {'pcc': 0.77, '其它指标以此类推': 33.1}, # fold 2
        }}
        # 存储的表格示意
        # epoch	1
        # 	fold1	fold2	fold3	fold4	avg±std
        # pcc	0.88	0.88	0.88	0.88	0.88 ±0
        # 其它指标	111.1	111.1	111.1	111.1	111.1 ±0
        # 其它指标	111.1	111.1	111.1	111.1	111.1 ±0
        # 其它指标	111.1	111.1	111.1	111.1	111.1 ±0
        self.result = {}

    # -----------------------------
    # 初始化 epoch
    # -----------------------------
    def _init_epoch(self, epoch):
        if str(epoch) not in self.result:
            self.result[str(epoch)] = {}

    def _format_value(self, key, value):
        # PCC：保留 3位小数
        if "pcc" in key:
            return round(value, 3)

        # HRV / delay 类 → 转 ms
        return round(value * 1000.0, 1)

    # -----------------------------
    # fold-level update
    # -----------------------------
    def get_result(self, ecg, pred_ecg, fold=0, epoch=None):
        self._init_epoch(epoch)

        pcc, lag = batch_max_pearson_corr(ecg, pred_ecg, max_lag=100)
        aligned_ecg, aligned_pred = align_ecg_radar(ecg, pred_ecg, lag)

        mt = self.ECG_physio_metrics.get_matrices(aligned_ecg, aligned_pred)

        if str(epoch) not in self.result:
            self.result[str(epoch)] = {}
        if fold not in self.result[str(epoch)]:
            self.result[str(epoch)][fold] = {}

        # -------------------------
        # 基础指标
        # -------------------------
        self.result[str(epoch)][fold]["pcc"] = round(float(pcc.mean().item()), 3)

        # -------------------------
        # ECG physio metrics
        # -------------------------
        for k, v in mt.items():
            if k.endswith('pct') or k.endswith('rate') or k == 'RMSE':
                self.result[str(epoch)][fold][k] = round(float(v), 2)
            else:
                self.result[str(epoch)][fold][k] = round(float(v) * 1000.0, 1)

        return self.result[str(epoch)][fold]

    # -----------------------------
    # 汇总 mean/std
    # -----------------------------
    def _aggregate_epoch(self, epoch):
        epoch = str(epoch)
        folds = self.result.get(epoch, {})

        if len(folds) == 0:
            return None, None

        df = pd.DataFrame.from_dict(folds, orient="index")
        df = df.sort_index()

        # metric × fold
        df = df.T

        avg = df.mean(axis=1)
        std = df.std(axis=1)

        return df, avg, std

    # -----------------------------
    # 生成 LaTeX 表格
    # -----------------------------
    def _to_latex(self, df):
        # 只保留表体
        return df.to_latex(index=False, float_format="%.4f")

    # -----------------------------
    # 保存结果
    # -----------------------------
    def save_result(self, epoch="best", res_dir='results'):
        epoch = str(epoch)

        if epoch not in self.result:
            print("No results to save.")
            return
        os.makedirs(res_dir, exist_ok=True)
        # =========================
        # 1. metric × fold
        # =========================
        df = pd.DataFrame.from_dict(self.result[epoch], orient="index")
        df = df.sort_index().T  # metric rows, fold cols

        # =========================
        # 2. mean ± std
        # =========================
        avg = df.mean(axis=1)
        std = df.std(axis=1)

        df["avg±std"] = [
            f"{avg[m]:.3f} ± {std[m]:.3f}" for m in df.index
        ]

        # =========================
        # 3. Excel 保存
        # =========================
        file_name = f"result_{self.model_name}_domain{self.domain}_{epoch}.xlsx"
        file_path = os.path.join(res_dir, file_name)
        df.to_excel(file_path)

        # =========================
        # 4. LaTeX（论文表体）
        # =========================
        latex_table = df.to_latex(float_format="%.3f")

        latex_file = file_name.replace(".xlsx", ".tex")
        latex_path = os.path.join(res_dir, latex_file)
        with open(latex_path, "w") as f:
            f.write(latex_table)

        print(f"[Done] saved to {file_name} + {latex_file}")







