import cv2

# Khởi tạo bộ ghi video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Chọn codec (ở đây sử dụng codec mp4v cho định dạng MP4)
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))  # Tên tệp, codec, fps, kích thước khung hình

# Khởi tạo camera
cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định, nếu có nhiều camera, thay đổi số 0 thành số tương ứng

while cap.isOpened():
    ret, frame = cap.read()  # Đọc frame từ camera

    if ret:
        # Ghi frame vào video
        out.write(frame)

        # Hiển thị frame
        cv2.imshow('Frame', frame)

        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Giải phóng các tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
