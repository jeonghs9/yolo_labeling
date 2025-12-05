import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox
)

from PyQt6.QtGui import QPixmap, QPainter, QPen
from PyQt6.QtCore import Qt, QRectF


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image = None          # QPixmap
        self.img_w = 0
        self.img_h = 0

        # boxes: list of (cls_id, x1, y1, x2, y2) in pixel coords
        self.boxes = []
        self.drawing = False
        self.start_pos = None
        self.current_box = None    # (x1, y1, x2, y2)

        # scaling / zoom
        self.base_scale = 1.0
        self.zoom_factor = 1.0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # panning (Ctrl + 드래그)
        self.panning = False
        self.pan_start = None      # QPointF
        self.pan_dx = 0.0
        self.pan_dy = 0.0

        # current class id (0~9)
        self.current_class = 0

        # 컨트롤 제트
        self.undo_stack = []

        # 클래스별 색상 매핑
        self.class_colors = {
            0: Qt.GlobalColor.yellow,
            1: Qt.GlobalColor.green,
            2: Qt.GlobalColor.red,
            3: Qt.GlobalColor.blue,
            # 4: Qt.GlobalColor.cyan,
            # 5: Qt.GlobalColor.magenta,
            # 6: Qt.GlobalColor.white,
            # 7: Qt.GlobalColor.gray,
            # 8: Qt.GlobalColor.darkRed,
            # 9: Qt.GlobalColor.darkGreen,
        }

    def set_image(self, pixmap: QPixmap | None):
        self.image = pixmap
        if pixmap is not None:
            self.img_w = pixmap.width()
            self.img_h = pixmap.height()
        else:
            self.img_w = self.img_h = 0
        # 이미지 바뀌면 줌/패닝 리셋
        self.zoom_factor = 1.0
        self.pan_dx = 0.0
        self.pan_dy = 0.0
        self.update()

    def set_boxes(self, boxes):
        """
        boxes: list of (cls_id, x1, y1, x2, y2) in pixel coords
        """
        self.boxes = boxes
        self.update()

    def get_boxes(self):
        return self.boxes

    def set_current_class(self, cls_id: int):
        self.current_class = int(cls_id)

    def undo(self):
        """
        Ctrl+Z → 마지막 작업(박스 추가)을 되돌리기
        """
        if not self.undo_stack:
            return

        action, box = self.undo_stack.pop()

        if action == "add":
            # 마지막 생성된 박스를 제거
            if box in self.boxes:
                self.boxes.remove(box)

        self.update()


    def mousePressEvent(self, event):
        if self.window():
            self.window().setFocus()

        if self.image is None:
            return

        # 위치 변환
        x, y = self._map_to_image(event.position())
        if x is None:
            return

        # 우클릭은 삭제 그대로 유지
        if event.button() == Qt.MouseButton.RightButton:
            removed = False
            for i in range(len(self.boxes) - 1, -1, -1):
                cls_id, x1, y1, x2, y2 = self.boxes[i]
                if (x1 <= x <= x2) and (y1 <= y <= y2):
                    del self.boxes[i]
                    removed = True
                    break
            if removed:
                self.update()
            return

        # 좌클릭 → 두 번 클릭 방식
        if event.button() == Qt.MouseButton.LeftButton:
            # 첫 클릭 (시작점)
            if not self.drawing:
                self.start_pos = (x, y)
                self.drawing = True
                self.current_box = None
            # 두 번째 클릭 (완료)
            else:
                x1, y1 = self.start_pos
                x2, y2 = x, y
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])

                new_box = (self.current_class, x_min, y_min, x_max, y_max)
                self.boxes.append(new_box)

                # undo 스택에 추가
                self.undo_stack.append(("add", new_box))

                # 초기화
                self.drawing = False
                self.start_pos = None
                self.current_box = None

                self.update()


    def mouseMoveEvent(self, event):
        # 드래그 기반 패닝 그대로 유지
        if self.panning and self.pan_start is not None:
            delta = event.position() - self.pan_start
            self.pan_start = event.position()
            self.pan_dx += delta.x()
            self.pan_dy += delta.y()
            self.update()
            return

        # 첫 클릭 이후 마우스를 움직이면 박스 미리보기
        if self.drawing and self.start_pos is not None:
            x, y = self._map_to_image(event.position())
            if x is None:
                return
            x1, y1 = self.start_pos
            self.current_box = (x1, y1, x, y)
            self.update()


    def mouseReleaseEvent(self, event):
        # 패닝 종료만 유지
        if event.button() == Qt.MouseButton.LeftButton and self.panning:
            self.panning = False
            self.pan_start = None


        def wheelEvent(self, event):
            # 이미지가 없으면 줌 동작 안 함
            if self.image is None or self.img_w == 0 or self.img_h == 0:
                return

            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor *= 1.1
            elif delta < 0:
                self.zoom_factor /= 1.1

        # 줌 한계 설정
        self.zoom_factor = max(0.2, min(self.zoom_factor, 10.0))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        if self.image is not None:
            label_w = self.width()
            label_h = self.height()

            if self.img_w == 0 or self.img_h == 0:
                return

            # 기본 스케일 (전체 이미지가 보이도록)
            scale_x = label_w / self.img_w
            scale_y = label_h / self.img_h
            self.base_scale = min(scale_x, scale_y)

            # 줌 반영한 최종 스케일
            self.scale = self.base_scale * self.zoom_factor

            draw_w = self.img_w * self.scale
            draw_h = self.img_h * self.scale

            # 기본은 중앙 정렬, 여기에 pan_dx/pan_dy 더함
            base_off_x = (label_w - draw_w) / 2
            base_off_y = (label_h - draw_h) / 2

            self.offset_x = base_off_x + self.pan_dx
            self.offset_y = base_off_y + self.pan_dy

            target_rect = QRectF(
                self.offset_x, self.offset_y,
                draw_w, draw_h
            )
            source_rect = QRectF(self.image.rect())
            painter.drawPixmap(target_rect, self.image, source_rect)

            # 박스들 그리기
            for cls_id, x1, y1, x2, y2 in self.boxes:
                color = self.class_colors.get(cls_id, Qt.GlobalColor.red)
                pen = QPen(color)
                pen.setWidth(2)
                painter.setPen(pen)

                rx1 = self.offset_x + x1 * self.scale
                ry1 = self.offset_y + y1 * self.scale
                rx2 = self.offset_x + x2 * self.scale
                ry2 = self.offset_y + y2 * self.scale
                painter.drawRect(QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1))

            # 드래그 중인 박스 (현재 클래스 색으로)
            if self.current_box is not None:
                x1, y1, x2, y2 = self.current_box
                color = self.class_colors.get(self.current_class, Qt.GlobalColor.white)
                pen = QPen(color)
                pen.setWidth(2)
                painter.setPen(pen)

                rx1 = self.offset_x + x1 * self.scale
                ry1 = self.offset_y + y1 * self.scale
                rx2 = self.offset_x + x2 * self.scale
                ry2 = self.offset_y + y2 * self.scale
                painter.drawRect(QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1))

    def _map_to_image(self, pos):
        """
        라벨 좌표(QPointF) → 원본 이미지 좌표(float)
        """
        if self.image is None or self.scale == 0:
            return None, None
        x = (pos.x() - self.offset_x) / self.scale
        y = (pos.y() - self.offset_y) / self.scale
        if x < 0 or y < 0 or x > self.img_w or y > self.img_h:
            return None, None
        return x, y


def yolo_to_boxes(label_path, img_w, img_h):
    """
    YOLO 포맷 txt → [(cls_id, x1, y1, x2, y2), ...]
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, w, h = parts
            cls_id = int(cls_id)
            xc = float(xc) * img_w
            yc = float(yc) * img_h
            w = float(w) * img_w
            h = float(h) * img_h
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def boxes_to_yolo(boxes, img_w, img_h):
    """
    [(cls_id, x1, y1, x2, y2), ...] → YOLO 포맷 문자열
    """
    lines = []
    for cls_id, x1, y1, x2, y2 in boxes:
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Labeling Tool (PyQt6)")

        self.image_dir = ""
        self.label_dir = ""
        self.image_files: list[str] = []
        self.current_index = -1

        # 현재 클래스 id (0~9)
        self.current_class = 0

        # 이미지별 임시 박스 캐시: {img_name: boxes}
        self.label_cache = {}

        # 라벨을 수정/확인한 이미지 집합(체크 표시용)
        self.modified_images = set()

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 메인 윈도우가 키보드 포커스를 받을 수 있게 설정
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 왼쪽: 폴더 버튼 + 현재 클래스 표시 + 이미지 리스트 + 저장 버튼
        left_layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_list_changed)

        btn_img_dir = QPushButton("이미지 폴더 선택")
        btn_img_dir.clicked.connect(self.select_image_dir)
        btn_lbl_dir = QPushButton("라벨 폴더 선택")
        btn_lbl_dir.clicked.connect(self.select_label_dir)

        self.class_label = QLabel(f"현재 클래스: {self.current_class}")

        self.clear_button = QPushButton("현재 이미지 라벨 삭제")
        self.clear_button.clicked.connect(self.clear_current_labels)

        self.save_button = QPushButton("현재 이미지 저장")
        self.save_button.clicked.connect(self.save_current_image)

        left_layout.addWidget(btn_img_dir)
        left_layout.addWidget(btn_lbl_dir)
        left_layout.addWidget(self.class_label)
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(self.clear_button)
        left_layout.addWidget(self.save_button)
        

        # 중앙: 이미지 라벨 위젯
        self.image_label = ImageLabel()
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.set_current_class(self.current_class)

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.image_label, 4)

        self.setCentralWidget(main_widget)
        self.resize(1400, 800)

    def keyPressEvent(self, event):
        key = event.key()

        # 다음/이전 이미지
        if key == Qt.Key.Key_Right:
            self.next_image()
        elif key == Qt.Key.Key_Left:
            self.prev_image()
        # 숫자 키로 클래스 선택 (0~9)
        elif Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            cls_id = key - Qt.Key.Key_0
            self.current_class = cls_id
            self.image_label.set_current_class(cls_id)
            self.class_label.setText(f"현재 클래스: {cls_id}")
        # Ctrl + Z → Undo
        elif (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_Z:
            self.image_label.undo()

        else:
            super().keyPressEvent(event)

    def select_image_dir(self):
        d = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if d:
            self.image_dir = d
            self.load_image_list()

    def select_label_dir(self):
        d = QFileDialog.getExistingDirectory(self, "라벨 폴더 선택")
        if d:
            self.label_dir = d

            # 현재 이미지 캐시 제거
            if self.image_files and self.current_index != -1:
                img_name = self.image_files[self.current_index]
                if img_name in self.label_cache:
                    del self.label_cache[img_name]

            # 강제로 라벨 포함해 다시 로딩
            if self.image_files and self.current_index != -1:
                self.load_current_image()

    def load_image_list(self):
        if not self.image_dir:
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = [
            f for f in sorted(os.listdir(self.image_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]

        # 새 이미지 폴더 선택하면 캐시/체크 상태 초기화
        self.label_cache.clear()
        self.modified_images.clear()

        self.list_widget.clear()
        self.list_widget.addItems(self.image_files)

        # 리스트 아이템을 체크 가능하게 만들기
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)

        if self.image_files:
            self.current_index = 0
            self.list_widget.setCurrentRow(0)
            self.load_current_image()

    def on_list_changed(self, row: int):
        if row < 0 or row >= len(self.image_files):
            return

        # 1) 현재 보이던 이미지(이전 인덱스)에 대한 캐시 저장
        if 0 <= self.current_index < len(self.image_files):
            self.update_cache_for_current_image()

        # 2) 인덱스 갱신
        self.current_index = row

        # 3) 새 이미지 로드
        self.load_current_image()

    def current_image_path(self):
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return None
        return os.path.join(self.image_dir, self.image_files[self.current_index])

    def current_label_path(self):
        if not self.label_dir or self.current_index < 0:
            return None
        img_name = self.image_files[self.current_index]
        base, _ = os.path.splitext(img_name)
        return os.path.join(self.label_dir, base + ".txt")

    def update_cache_for_current_image(self):
        """현재 화면에 그려진 박스를 캐시에 저장 (파일에는 안 씀)"""
        if self.current_index < 0 or not self.image_files:
            return

        img_name = self.image_files[self.current_index]
        boxes = self.image_label.get_boxes()

        # 얕은 복사해서 보관
        self.label_cache[img_name] = list(boxes)

        # 수정된/확인된 이미지로 표시
        self.modified_images.add(img_name)
        item = self.list_widget.item(self.current_index)
        if item is not None:
            item.setCheckState(Qt.CheckState.Checked)
    
    def clear_current_labels(self):
        """
        현재 이미지의 바운딩박스를 모두 제거하는 버튼 동작
        - 화면에서 박스 삭제
        - 캐시에도 빈 리스트로 반영
        - 수정된 이미지로 체크 표시 유지
        """
        if self.current_index < 0 or not self.image_files:
            return

        # 선택 확인 팝업 (원하면 빼도 됨)
        reply = QMessageBox.question(
            self,
            "라벨 삭제 확인",
            "현재 이미지의 모든 라벨을 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # 1) 화면에서 박스 모두 제거
        self.image_label.set_boxes([])

        # 2) 캐시/체크 상태 업데이트
        self.update_cache_for_current_image()


    def load_current_image(self):
        img_path = self.current_image_path()
        if not img_path:
            return
        pixmap = QPixmap(img_path)
        self.image_label.set_image(pixmap)

        img_name = self.image_files[self.current_index]
        boxes = []

        # 이미 수정한 이미지라면 체크 표시 유지
        item = self.list_widget.item(self.current_index)
        if item is not None:
            if img_name in self.modified_images:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                # 필요하다면 수정 안 된 건 Unchecked로
                # item.setCheckState(Qt.CheckState.Unchecked)
                pass

        # 1순위: 캐시
        if img_name in self.label_cache:
            boxes = self.label_cache[img_name]
        else:
            lbl_path = self.current_label_path()
            if lbl_path and os.path.exists(lbl_path):
                boxes = yolo_to_boxes(lbl_path, self.image_label.img_w, self.image_label.img_h)

        self.image_label.set_boxes(boxes)


    def save_current_image(self):
        """
        Save 버튼 클릭 시:
        - 현재 이미지 포함, 지금까지 수정/체크된 모든 이미지의 라벨을
          txt 파일로 덮어쓰기 저장
        - 몇 개 저장됐는지 팝업으로 알려줌
        """
        if not self.label_dir:
            return

        # 현재 화면에 보이는 것도 캐시에 반영
        self.update_cache_for_current_image()

        if not self.modified_images:
            QMessageBox.information(self, "저장", "수정된 이미지가 없습니다.")
            return

        os.makedirs(self.label_dir, exist_ok=True)

        saved = 0
        for img_name in sorted(self.modified_images):
            boxes = self.label_cache.get(img_name, [])

            # 각 이미지의 원본 크기 필요 → 직접 로드
            img_path = os.path.join(self.image_dir, img_name)
            pixmap = QPixmap(img_path)
            if pixmap.isNull():
                continue
            img_w = pixmap.width()
            img_h = pixmap.height()
            if img_w == 0 or img_h == 0:
                continue

            base, _ = os.path.splitext(img_name)
            lbl_path = os.path.join(self.label_dir, base + ".txt")

            yolo_txt = boxes_to_yolo(boxes, img_w, img_h)
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write(yolo_txt)
            saved += 1

        QMessageBox.information(
            self,
            "저장 완료",
            f"체크된 이미지 {len(self.modified_images)}개 중 {saved}개를 저장했습니다."
        )

    def next_image(self):
        if not self.image_files:
            return
        if self.current_index < len(self.image_files) - 1:
            # 인덱스는 여기서 건드리지 않고, 선택만 변경
            self.list_widget.setCurrentRow(self.current_index + 1)

    def prev_image(self):
        if not self.image_files:
            return
        if self.current_index > 0:
            self.list_widget.setCurrentRow(self.current_index - 1)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "프로그램 종료",
            "정말 종료하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()   # 진짜 닫기
        else:
            event.ignore()   # 닫기 취소

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LabelingTool()
    win.show()
    sys.exit(app.exec())
