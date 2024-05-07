import os
import glob

glob_pattern = os.path.join("calibdir_*", "iter*", "results-cam-calibreconstruction.txt")
predicted_calib_files = sorted(glob.glob(glob_pattern))

model_results_dict = {}

for predicted_calib_file in predicted_calib_files:
    model_name = predicted_calib_file.split(os.sep)[0].split('calibdir_')[-1]
    with open(predicted_calib_file, 'r') as f:
        lines = f.readlines()
        projection_line = lines[6]
        predicted_camera_params = projection_line.split('projection: [')[-1].split('] +- [')[0].split()
        fx, fy, cx, cy = [float(param) for param in predicted_camera_params]
        distortion_line = lines[5]
        predicted_distortion_params = distortion_line.split('distortion: [')[-1].split('] +- [')[0].split()
        k1, k2 = [float(param) for param in predicted_distortion_params][:2]
        if model_name not in model_results_dict:
            model_results_dict[model_name] = []
        model_results_dict[model_name].append((fx, fy, cx, cy, k1, k2))


gt_params_file = "gt_calib_params.txt"
with open(gt_params_file, 'r') as f:
    lines = f.readlines()
    gt_params = [float(p) for p in lines[0].split()[:6]]

for model_name in model_results_dict:
    model_average_preds = [sum(pred) / len(pred) for pred in zip(*model_results_dict[model_name])]
    model_abs_errors = [abs(gt - pred) for gt, pred in zip(gt_params, model_average_preds)]
    model_percentage_errors = [abs_error / gt for gt, abs_error in zip(gt_params, model_abs_errors)]
    model_mape = 100 * sum(model_percentage_errors) / len(model_percentage_errors)
    print(f'{model_name} MAPE: {model_mape:.2f}%')
