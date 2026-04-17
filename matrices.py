import numpy as np
import neurokit2 as nk


class ECGPhysioMetrics:
    def __init__(
            self,
            sampling_rate=200,
            event_types=("R", "P", "Q", "S", "T"),
            hrv_metrics=("SDNN", "RMSSD", "pNN50")
    ):
        super().__init__()
        self.fs = sampling_rate
        self.event_types = event_types
        self.hrv_metrics = hrv_metrics

    def _extract_events(self, ecg):
        events = {e: np.array([]) for e in self.event_types}

        try:
            ecg_clean = nk.ecg_clean(ecg, sampling_rate=self.fs)

            _, info = nk.ecg_peaks(ecg_clean, sampling_rate=self.fs)
            r_peaks = np.array(info.get("ECG_R_Peaks", []), dtype=float)

            r_peaks = r_peaks[~np.isnan(r_peaks)]
            events["R"] = r_peaks

            if any(e in self.event_types for e in ["P", "Q", "S", "T"]) and len(r_peaks) > 0:
                try:
                    _, delineate = nk.ecg_delineate(
                        ecg_clean,
                        r_peaks,
                        sampling_rate=self.fs,
                        method="dwt"
                    )
                except Exception as ex:
                    print("delineate failed:", ex)
                    delineate = {}

                for e in self.event_types:
                    if e == "R":
                        continue

                    key = f"ECG_{e}_Peaks"

                    if key in delineate:
                        vals = np.array(delineate[key], dtype=float)
                        vals = vals[~np.isnan(vals)]  # 关键
                        events[e] = vals
                    else:
                        events[e] = np.array([])

            return events

        except Exception as ex:
            print(ex)
            return {e: np.array([]) for e in self.event_types}

    def _event_delay(self, pred_idx, target_idx, rpeaks=None, thr=0.1):
        if len(pred_idx) == 0 or len(target_idx) == 0:
            return {
                "mae_s": None,
                "std_s": None,
                "mae_pct": None,
                "std_pct": None,
                "median_pct": None,
                "detect_rate": 0
            }

        pred_idx = np.array(pred_idx, dtype=float)
        target_idx = np.array(target_idx, dtype=float)

        # 去 NaN
        pred_idx = pred_idx[~np.isnan(pred_idx)]
        target_idx = target_idx[~np.isnan(target_idx)]

        if len(pred_idx) == 0 or len(target_idx) == 0:
            return {
                "mae_s": None,
                "std_s": None,
                "mae_pct": None,
                "std_pct": None,
                "median_pct": None,
                "detect_rate": 0
            }

        # RR interval（sample）
        rr_intervals = None
        if rpeaks is not None and len(rpeaks) > 1:
            rpeaks = np.array(rpeaks, dtype=float)
            rr_intervals = np.diff(rpeaks)

        i = 0
        delays_s = []  # 秒
        delays_pct = []  # 归一化 %
        matched = 0
        thr_samples = thr * self.fs

        for t in target_idx:
            # 找最近 pred
            while i < len(pred_idx) - 1 and abs(pred_idx[i + 1] - t) < abs(pred_idx[i] - t):
                i += 1

            closest = pred_idx[i]
            diff = closest - t

            if abs(diff) <= thr_samples:
                matched += 1

                # 👉 秒误差
                delay_s = diff / self.fs
                delays_s.append(delay_s)

                # 👉 归一化误差
                if rr_intervals is not None:
                    r_idx = np.argmin(np.abs(rpeaks - t))

                    if r_idx == 0:
                        rr = rr_intervals[0]
                    elif r_idx >= len(rr_intervals):
                        rr = rr_intervals[-1]
                    else:
                        rr = rr_intervals[r_idx]

                    if rr > 0:
                        delay_pct = abs(diff) / rr
                        delays_pct.append(delay_pct)

        if len(delays_s) == 0:
            return {
                "mae_s": None,
                "std_s": None,
                "mae_pct": None,
                "std_pct": None,
                "median_pct": None,
                "detect_rate": 0
            }

        delays_s = np.array(delays_s)
        delays_pct = np.array(delays_pct) if len(delays_pct) > 0 else None

        results = {
            "mae_s": np.mean(np.abs(delays_s)),
            "std_s": np.std(delays_s),

            "detect_rate": matched / len(target_idx)
        }

        # ✅ 归一化指标（%）
        if delays_pct is not None and len(delays_pct) > 0:
            results.update({
                "mae_pct": np.mean(delays_pct) * 100,
                "std_pct": np.std(delays_pct) * 100,
                "median_pct": np.median(delays_pct) * 100
            })
        else:
            results.update({
                "mae_pct": None,
                "std_pct": None,
                "median_pct": None
            })

        return results

    # -----------------------------
    # RR
    # -----------------------------
    def _get_rr(self, rpeaks):
        if len(rpeaks) < 2:
            return None
        return np.diff(rpeaks) / self.fs

    # -----------------------------
    # HRV
    # -----------------------------
    def _hrv(self, rr):
        if rr is None or len(rr) < 2:
            return {m: None for m in self.hrv_metrics}

        res = {}

        if "MeanRR" in self.hrv_metrics:
            res["MeanRR"] = np.mean(rr)

        if "SDNN" in self.hrv_metrics:
            res["SDNN"] = np.std(rr, ddof=1)

        if "RMSSD" in self.hrv_metrics:
            res["RMSSD"] = np.sqrt(np.mean(np.diff(rr) ** 2))

        if "pNN50" in self.hrv_metrics:
            diff_rr = np.abs(np.diff(rr))
            res["pNN50"] = np.mean(diff_rr > 0.05)  # 50 ms = 0.05 s

        return res

    # -----------------------------
    # forward（统计）
    # -----------------------------
    def get_matrices(self, pred_ecg, target_ecg):
        pred_ecg = pred_ecg.detach().cpu().numpy()
        target_ecg = target_ecg.detach().cpu().numpy()

        if len(pred_ecg.shape) == 3:
            pred_ecg = pred_ecg.squeeze(1)
        if len(target_ecg.shape) == 3:
            target_ecg = target_ecg.squeeze(1)

        results = {
            "event": {e: [] for e in self.event_types},
            "hrv_diff": {m: [] for m in self.hrv_metrics},
            "hrv_pred": {m: [] for m in self.hrv_metrics},
            "hrv_target": {m: [] for m in self.hrv_metrics},
        }

        # =========================
        # per-batch computation
        # =========================
        rmse_list = []

        def normalize_to_1_1(signal):
            """将信号归一化到[-1, 1]范围"""
            min_val = np.min(signal)
            max_val = np.max(signal)
            # 避免除以零
            if max_val - min_val == 0:
                return np.zeros_like(signal)
            # 归一化到[-1, 1]
            normalized = 2 * (signal - min_val) / (max_val - min_val) - 1
            return normalized

        for b in range(pred_ecg.shape[0]):
            pred_events = self._extract_events(pred_ecg[b])
            target_events = self._extract_events(target_ecg[b])

            # -------- Event (NEW FORMAT) --------
            for e in self.event_types:
                stats = self._event_delay(
                    pred_events.get(e, []),
                    target_events.get(e, []),
                    rpeaks=target_events.get("R", []),  # ⭐关键：用于RR归一化
                )

                if stats is not None:
                    results["event"][e].append(stats)

            # -------- HRV --------
            rr_pred = self._get_rr(pred_events.get("R", []))
            rr_target = self._get_rr(target_events.get("R", []))

            pred_hrv = self._hrv(rr_pred)
            target_hrv = self._hrv(rr_target)

            for m in self.hrv_metrics:
                if pred_hrv[m] is not None and target_hrv[m] is not None:
                    diff = pred_hrv[m] - target_hrv[m]
                    results["hrv_diff"][m].append(diff)
                    results["hrv_target"][m].append(target_hrv[m])
                    results["hrv_pred"][m].append(pred_hrv[m])

            # -------- RMSE --------
            rmse = np.sqrt(np.mean((normalize_to_1_1(pred_ecg[b]) - normalize_to_1_1(target_ecg[b])) ** 2))
            rmse_list.append(rmse)

        # =========================
        # aggregation
        # =========================
        summary = {}
        summary["RMSE"] = np.mean(rmse_list)

        # -------- Event summary --------
        for e in self.event_types:
            if len(results["event"][e]) > 0:
                stats_list = results["event"][e]

                mae_s = [s["mae_s"] for s in stats_list if s["mae_s"] is not None]
                mae_pct = [s["mae_pct"] for s in stats_list if s["mae_pct"] is not None]
                median_pct = [s["median_pct"] for s in stats_list if s["median_pct"] is not None]
                detect = [s["detect_rate"] for s in stats_list]

                summary[f"{e}_mae_s"] = np.mean(mae_s) if len(mae_s) > 0 else 0.0
                summary[f"{e}_mae_pct"] = np.mean(mae_pct) if len(mae_pct) > 0 else 0.0
                summary[f"{e}_median_pct"] = np.median(median_pct) if len(median_pct) > 0 else 0.0
                summary[f"{e}_detect_rate"] = np.mean(detect) if len(detect) > 0 else 0.0
            else:
                summary[f"{e}_mae_s"] = 0.0
                summary[f"{e}_mae_pct"] = 0.0
                summary[f"{e}_median_pct"] = 0.0
                summary[f"{e}_detect_rate"] = 0.0

        # -------- HRV --------
        for m in self.hrv_metrics:
            if len(results["hrv_diff"][m]) > 0:
                vals_diff = np.array(results["hrv_diff"][m])
                vals_pred = np.array(results["hrv_pred"][m])
                vals_target = np.array(results["hrv_target"][m])

                summary[f"{m}_diff_mean"] = vals_diff.mean()
                summary[f"{m}_diff_std"] = vals_diff.std()

                summary[f"{m}_pred_mean"] = vals_pred.mean()
                summary[f"{m}_pred_std"] = vals_pred.std()

                summary[f"{m}_target_mean"] = vals_target.mean()
                summary[f"{m}_target_std"] = vals_target.std()
            else:
                summary[f"{m}_diff_mean"] = 0.0
                summary[f"{m}_diff_std"] = 0.0
                summary[f"{m}_pred_mean"] = 0.0
                summary[f"{m}_pred_std"] = 0.0
                summary[f"{m}_target_mean"] = 0.0
                summary[f"{m}_target_std"] = 0.0

        return summary
