import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

class SAREvaluator:
    """
    Evaluation suite for SAR-specific image quality metrics.

    Focuses on the impulse response frequency (IRF) and radiometric consistency of reconstructed single-look complex
    (SLC) data.

    Parameters
    ----------
    oversampling : int, default=16
        The factor used for cubic interpolation when calculating 3 dB bandwidth.
    global_max : float, default=4257
        The normalization constant used to scale tensors back to physical units.
    """
    def __init__(self, oversampling=16, global_max=4257):
        self.oversampling = oversampling
        self.global_max = global_max

    def to_complex(self, tensor):
        """
        Converts interleaved real / imaginary PyTorch tensors to NumPy complex64.

        Parameters
        ----------
        tensor : torch.Tensor
            A tensor of shape (Batch, Channels, Height, Width) where channels 0 and 1 represent Real and Imaginary
            components.

        Returns
        -------
        np.ndarray (complex64)
            The physically scaled complex-valued array.
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        real = tensor[:, 0::2, ...].detach().cpu()
        imag = tensor[:, 1::2, ...].detach().cpu()
        return (torch.complex(real, imag) * self.global_max).numpy()

    def calculate_3db_width(self, signal_1d):
        """
        Estimates the 3 dB mainlobe width of a 1D SAR signal.

        Applies cubic interpolation to the magnitude profile to find the width where the power drops to 50% (-3 dB) of
        the peak.

        Parameters
        ----------
        signal_1d : np.ndarray (complex64)
            A 1D complex profile along azimuth or range.

        Returns
        -------
        float
            The width in pixels at the -3 dB point. Returns np.nan if the profile is invalid.
        """
        mag = np.abs(signal_1d)
        if np.max(mag) == 0: return np.nan
        mag_db = 20 * np.log10(mag / (np.max(mag) + 1e-9))

        x = np.arange(len(mag_db))
        x_new = np.linspace(0, len(mag_db)-1, len(mag_db) * self.oversampling)
        f = interp1d(x, mag_db, kind='cubic')
        mag_os = f(x_new)

        above_3db = np.where(mag_os >= -3.0)[0]
        if len(above_3db) < 2:
            return np.nan

        width = (above_3db[-1] - above_3db[0]) / self.oversampling
        return width

    def get_irf_metrics(self, pred_complex, gt_complex):
        """
        Computes the impulse response function (IRF) metrics for azimuth and range.

        Extracts 1D profiles centered on the peak intensity of the ground truth and calculates the relative error.

        Parameters
        ----------
        pred_complex : np.ndarray (complex64)
            The reconstructed complex-valued SAR patch.
        gt_complex : np.ndarray (complex64)
            The ground truth complex valued SAR patch.

        Returns
        -------
        dict
            Dictionary containing 'azimuth_width_err', 'range_width_err', and predicted widths.
        """
        if gt_complex.ndim != 2:
            gt_complex = np.squeeze(gt_complex)
            pred_complex = np.squeeze(pred_complex)
        gt_mag = np.abs(gt_complex)
        idx_h, idx_w = np.unravel_index(np.argmax(gt_mag), gt_mag.shape)

        metrics = {}
        for dim_name, gt_prof, pred_prof in [
            ("azimuth", gt_complex[:, idx_w], pred_complex[:, idx_w]),
            ("range", gt_complex[idx_h, :], pred_complex[idx_h, :])
        ]:
            gt_w = self.calculate_3db_width(gt_prof)
            pred_w = self.calculate_3db_width(pred_prof)

            h_key = "h1_azimuth_err" if dim_name == "azimuth" else "h2_range_err"
            metrics[h_key] = abs(pred_w - gt_w) / gt_w if gt_w >0 else np.nan

        return metrics

    def get_radiometric_metrics(self, pred_complex, gt_complex):
        """
        Calculates intensity preservation and equivalent number of looks (ENL).

        ENL is used to measure speckle reduction and radiometric resolution consistency in the reconstruction.

        Parameters
        ----------
        pred_complex : np.ndarray (complex64)
            The reconstructed complex-valued SAR patch.
        gt_complex : np.ndarray (complex64)
            The ground truth complex-valued SAR patch.

        Returns
        -------
        dict
            Dictionary containing 'intensity_err', 'enl_err', and 'enl_val'.
        """
        pred_int = np.abs(pred_complex) ** 2
        gt_int = np.abs(gt_complex) ** 2

        mean_pred = np.mean(pred_int)
        mean_gt = np.mean(gt_int)
        intensity_err = abs(mean_pred - mean_gt) / mean_gt if mean_gt > 0 else np.nan

        enl_pred = (np.mean(pred_int) ** 2) / (np.var(pred_int) + 1e-9)
        enl_gt = (np.mean(gt_int) ** 2) / (np.var(gt_int) + 1e-9)
        enl_err = abs(enl_pred - enl_gt) / enl_gt if enl_gt > 0 else np.nan

        return {
            "h3_intensity_err": intensity_err,
            "h3_enl_err": enl_err,
        }

    def evaluate_hypotheses(self, pred, gt):
        """
        Evaluates a single batch for hypotheses H1 to H3 and aggregates all SAR-specific metrics for a single
        prediction-target pair.

        Parameters
        ----------
        pred : torch.Tensor
            Normalized model output tensor.
        gt : torch.Tensor
            Normalized ground truth tensor.

        Returns
        -------
        dict
            A merged dictionary of IRD and radiometric evaluation metrics.
        """
        pred_c = self.to_complex(pred)
        gt_c = self.to_complex(gt)

        batch_results = []
        for b in range(pred_c.shape[0]):
            for s in range(pred_c.shape[1]):
                p_sample = pred_c[b][s]
                g_sample = gt_c[b][s]

                irf = self.get_irf_metrics(p_sample, g_sample)
                radio = self.get_radiometric_metrics(p_sample, g_sample)

                batch_results.append({**irf, **radio})

        return batch_results


def check_norm(tensor, name):
    mag_part = tensor[0::2]
    phase_part = tensor[1::2]

    print(f"--- {name} Normalization Check ---")
    print(f"Shape: {tensor.shape}")
    print(f"Mag   | Min: {mag_part.min():.4f}, Max: {mag_part.max():.4f}, Mean: {mag_part.mean():.4f}")
    print(f"Phase | Min: {phase_part.min():.4f}, Max: {phase_part.max():.4f}, Mean: {phase_part.mean():.4f}")

    assert tensor.max() <= 1.0, f"{name} exceeds 1.0"
    assert tensor.min() >= -1.0, f"{name} is below -1.0"
    print(f"Status: PASSED\n")