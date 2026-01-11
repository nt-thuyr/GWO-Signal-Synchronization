import numpy as np

class ChaoticMap:
    def __init__(self, map_type="logistic", start_val=0.7):
        self.map_type = map_type
        self.val = start_val
        
        if map_type == "logistic" and self.val in [0, 0.25, 0.5, 0.75, 1]:
            print("Cảnh báo: Giá trị khởi đầu không tốt. Đã tự động đổi sang 0.7")
            self.val = 0.7
            
    def next(self):
        if self.map_type == "logistic":
            # Công thức: x_new = 4 * x * (1 - x)
            self.val = 4 * self.val * (1 - self.val)
            
        elif self.map_type == "tent":
            if self.val < 0.5:
                self.val = 2 * self.val
            else:
                self.val = 2 * (1 - self.val)
                
        return self.val

    def next_vector(self, dim):
        # Trả về một mảng (vector) các số hỗn loạn
        vector = []
        for _ in range(dim):
            # Gọi hàm next() dim lần để lấp đầy vector
            vector.append(self.next())
        return np.array(vector)    