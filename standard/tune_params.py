import numpy as np
import time
from gwo_standard import GWO
from problem_model import SignalSyncProblem

# 1. Định nghĩa các danh sách tham số cần thử
wolves_list = [30, 50, 100]
iters_list = [50, 100, 200]

# Tạo một bài toán cố định để test cho công bằng
# (Cùng một tín hiệu, cùng đáp án, xem cấu hình nào giải tốt hơn)
problem = SignalSyncProblem(num_symbols=100, Fs=20, noise_power=0.1)
lb, ub = problem.get_bounds()
dim = 2

print(f"{'Wolves':<10} | {'Iters':<10} | {'Time(s)':<10} | {'Error Tau':<10} | {'Error Phi':<10}")
print("-" * 65)

# 2. Vòng lặp thử nghiệm (Nested Loop)
for pop_size in wolves_list:
    for max_iter in iters_list:
        
        # --- BẮT ĐẦU ĐO THỜI GIAN ---
        start_time = time.time()
        
        # TODO 1: Khởi tạo GWO với pop_size và max_iter hiện tại
        my_gwo = GWO(problem.fitness_function, lb, ub, 2, pop_size, max_iter)
        
        # TODO 2: Chạy optimize()
        best_pos, best_score, _ = my_gwo.optimize()
        
        # --- KẾT THÚC ĐO THỜI GIAN ---
        end_time = time.time()
        run_time = end_time - start_time
        
        # TODO 3: Tính sai số
        # Đáp án Sói đoán
        tau_est = int(best_pos[0])
        phi_est = best_pos[1]
        # Đáp án thật nằm trong biến problem
        # Sai số = |Đáp án thật - Đáp án Sói đoán|
        err_tau = abs(problem.tau_true - tau_est)
        err_phi = abs(problem.phi_true - phi_est)
        
        # In kết quả dạng bảng
        print(f"{pop_size:<10} | {max_iter:<10} | {run_time:<10.4f} | {err_tau:<10} | {err_phi:<10.4f}")