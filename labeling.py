import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QPolygonF, QCursor
from PyQt6.QtCore import Qt, QRectF, QPointF



# ==========================
#   ImageLabel 위젯
# ==========================
class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 이미지 상태
        self.image: QPixmap | None = None
        self.img_w = 0
        self.img_h = 0

        # 모드: "bbox" or "seg"
        self.mode = "bbox"

        # BBox 데이터: (cls, x1, y1, x2, y2) in pixel
        self.boxes: list[tuple[int, float, float, float, float]] = []
        self.box_start: tuple[float, float] | None = None
        self.box_preview_end: tuple[float, float] | None = None

        # Seg 데이터: (cls, [(x,y), ...]) in pixel
        self.polygons: list[tuple[int, list[tuple[float, float]]]] = []
        self.current_polygon: list[tuple[float, float]] = []
        self.seg_preview_point: tuple[float, float] | None = None

        # 줌/팬
        self.zoom_factor = 1.0
        self.base_scale = 1.0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.pan_dx = 0.0
        self.pan_dy = 0.0
        self.panning = False
        self.pan_start = None

        # 현재 클래스
        self.current_class = 0
        self.class_colors: dict[int, Qt.GlobalColor] = {
            0: Qt.GlobalColor.yellow,
            1: Qt.GlobalColor.green,
            2: Qt.GlobalColor.red,
            3: Qt.GlobalColor.blue,
        }

        self.setMouseTracking(True)

    # ----------------------------
    #   기본 설정/조회 함수
    # ----------------------------
    def set_image(self, pixmap: QPixmap | None):
        self.image = pixmap
        if pixmap is not None:
            self.img_w = pixmap.width()
            self.img_h = pixmap.height()
        else:
            self.img_w = self.img_h = 0

        # 뷰 리셋
        self.zoom_factor = 1.0
        self.pan_dx = self.pan_dy = 0.0
        self.box_start = None
        self.box_preview_end = None
        self.current_polygon = []
        self.seg_preview_point = None

        self.update()

    def set_current_class(self, cls_id: int):
        self.current_class = int(cls_id)

    def set_boxes(self, boxes):
        self.boxes = list(boxes)
        # 박스 설정 시 진행 중이던 것 초기화
        self.box_start = None
        self.box_preview_end = None
        self.update()

    def set_polygons(self, polygons):
        # 깊은 복사 수준까진 아니어도 list copy
        self.polygons = [(cls, list(pts)) for cls, pts in polygons]
        self.current_polygon = []
        self.seg_preview_point = None
        self.update()

    def get_boxes(self):
        return list(self.boxes)

    def get_polygons(self):
        return [(cls, list(pts)) for cls, pts in self.polygons]

    # ----------------------------
    #   마우스 이벤트
    # ----------------------------
    def mousePressEvent(self, event):
        if self.image is None:
            return

        mods = event.modifiers()
        pos = event.position()

        # Ctrl + 좌클릭 → 패닝 시작
        if (
            event.button() == Qt.MouseButton.LeftButton
            and (mods & Qt.KeyboardModifier.ControlModifier)
        ):
            self.panning = True
            self.pan_start = pos
            return

        x, y = self._map_to_image(pos)
        if x is None:
            return

        # 우클릭
        if event.button() == Qt.MouseButton.RightButton:
            if self.mode == "bbox":
                self._delete_bbox_at(x, y)
            else:
                self._delete_seg_point_at(x, y)
            return

        # 좌클릭
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "bbox":
                self._bbox_click(x, y)
            else:
                self._seg_add_point(x, y)

    def mouseDoubleClickEvent(self, event):
        # 세그 모드에서 더블클릭 → 폴리곤 닫기
        if (
            self.mode == "seg"
            and event.button() == Qt.MouseButton.LeftButton
            and len(self.current_polygon) >= 3
        ):
            self.finish_polygon()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning and self.pan_start is not None:
            delta = event.position() - self.pan_start
            self.pan_start = event.position()
            self.pan_dx += delta.x()
            self.pan_dy += delta.y()
            self.update()
            return

        if self.image is None:
            return

        x, y = self._map_to_image(event.position())
        if x is None:
            return

        # BBox 프리뷰
        if self.mode == "bbox" and self.box_start is not None:
            self.box_preview_end = (x, y)
            self.update()
            return

        # Seg 프리뷰: 마지막 점에서 마우스까지 라인
        if self.mode == "seg" and len(self.current_polygon) > 0:
            self.seg_preview_point = (x, y)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.panning:
            self.panning = False
            self.pan_start = None

    # ----------------------------
    #   휠 줌
    # ----------------------------
    def wheelEvent(self, event):
        if self.image is None or self.img_w == 0 or self.img_h == 0:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.1
        elif delta < 0:
            self.zoom_factor /= 1.1
        self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))
        self.update()

    # ----------------------------
    #   그리기
    # ----------------------------
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        if self.image is None or self.img_w == 0 or self.img_h == 0:
            return

        label_w = self.width()
        label_h = self.height()

        self.base_scale = min(label_w / self.img_w, label_h / self.img_h)
        self.scale = self.base_scale * self.zoom_factor

        draw_w = self.img_w * self.scale
        draw_h = self.img_h * self.scale

        self.offset_x = (label_w - draw_w) / 2 + self.pan_dx
        self.offset_y = (label_h - draw_h) / 2 + self.pan_dy

        # 이미지
        target_rect = QRectF(self.offset_x, self.offset_y, draw_w, draw_h)
        source_rect = QRectF(0, 0, self.img_w, self.img_h)
        painter.drawPixmap(target_rect, self.image, source_rect)

        # --------- Seg: 확정된 폴리곤 채우기 + 테두리 ---------
        for cls_id, pts in self.polygons:
            if len(pts) < 3:
                continue
            color = QColor(self.class_colors.get(cls_id, Qt.GlobalColor.red))
            fill_color = QColor(color.red(), color.green(), color.blue(), 80)  # 투명도 80

            qpoints = []
            for (x, y) in pts:
                px = self.offset_x + x * self.scale
                py = self.offset_y + y * self.scale
                qpoints.append(QPointF(px, py))

            poly = QPolygonF(qpoints)
            painter.setBrush(fill_color)
            painter.setPen(QPen(color, 2))
            painter.drawPolygon(poly)

        # --------- BBox 박스들 ---------
        for cls_id, x1, y1, x2, y2 in self.boxes:
            color = self.class_colors.get(cls_id, Qt.GlobalColor.red)
            pen = QPen(color, 2)
            painter.setPen(pen)

            rx1 = self.offset_x + x1 * self.scale
            ry1 = self.offset_y + y1 * self.scale
            rx2 = self.offset_x + x2 * self.scale
            ry2 = self.offset_y + y2 * self.scale
            painter.drawRect(QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1))

        # --------- BBox 프리뷰 박스 ---------
        if self.mode == "bbox" and self.box_start is not None and self.box_preview_end is not None:
            x1, y1 = self.box_start
            x2, y2 = self.box_preview_end
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            color = self.class_colors.get(self.current_class, Qt.GlobalColor.white)
            pen = QPen(color, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)

            rx1 = self.offset_x + x_min * self.scale
            ry1 = self.offset_y + y_min * self.scale
            rw = (x_max - x_min) * self.scale
            rh = (y_max - y_min) * self.scale
            painter.drawRect(QRectF(rx1, ry1, rw, rh))

        # --------- Seg: 현재 그리고 있는 폴리곤 선 ---------
        if self.mode == "seg" and len(self.current_polygon) > 0:
            color = self.class_colors.get(self.current_class, Qt.GlobalColor.white)
            pen = QPen(color, 2)
            painter.setPen(pen)

            # 클릭으로 만든 점들 사이 선
            for i in range(len(self.current_polygon) - 1):
                x1, y1 = self.current_polygon[i]
                x2, y2 = self.current_polygon[i + 1]
                sx1 = self.offset_x + x1 * self.scale
                sy1 = self.offset_y + y1 * self.scale
                sx2 = self.offset_x + x2 * self.scale
                sy2 = self.offset_y + y2 * self.scale
                painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

            # 마지막 점 → 마우스 위치 프리뷰
            if self.seg_preview_point is not None:
                x1, y1 = self.current_polygon[-1]
                x2, y2 = self.seg_preview_point
                sx1 = self.offset_x + x1 * self.scale
                sy1 = self.offset_y + y1 * self.scale
                sx2 = self.offset_x + x2 * self.scale
                sy2 = self.offset_y + y2 * self.scale
                painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

    # ----------------------------
    #   BBox 모드 로직
    # ----------------------------
    def _bbox_click(self, x, y):
        # 첫 클릭: 시작점
        if self.box_start is None:
            self.box_start = (x, y)
            self.box_preview_end = None
        # 두 번째 클릭: 박스 확정
        else:
            x1, y1 = self.box_start
            x2, y2 = x, y
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            # 너무 작은 박스는 무시(옵션)
            if (x_max - x_min) > 1 and (y_max - y_min) > 1:
                self.boxes.append((self.current_class, x_min, y_min, x_max, y_max))

            self.box_start = None
            self.box_preview_end = None
            self.update()

    def _delete_bbox_at(self, x, y):
        # 최근 것부터 탐색
        for i in range(len(self.boxes) - 1, -1, -1):
            cls_id, x1, y1, x2, y2 = self.boxes[i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                del self.boxes[i]
                break
        self.update()

    def undo_bbox(self):
        if self.box_start is not None:
            # 진행 중이던 박스 취소
            self.box_start = None
            self.box_preview_end = None
        elif self.boxes:
            self.boxes.pop()
        self.update()

    # ----------------------------
    #   Seg 모드 로직
    # ----------------------------
    def _seg_add_point(self, x, y):
        self.current_polygon.append((x, y))
        self.seg_preview_point = None
        self.update()

    def finish_polygon(self):
        if len(self.current_polygon) >= 3:
            self.polygons.append((self.current_class, list(self.current_polygon)))
        self.current_polygon = []
        self.seg_preview_point = None
        self.update()

    def _delete_seg_point_at(self, x, y, threshold=5.0):
        # threshold는 이미지 픽셀 기준
        thr2 = threshold * threshold

        # 먼저 점 삭제
        for poly_idx, (cls_id, pts) in enumerate(self.polygons):
            for i, (px, py) in enumerate(pts):
                dx = x - px
                dy = y - py
                if dx * dx + dy * dy <= thr2:
                    # 점 삭제
                    del pts[i]
                    if len(pts) < 3:
                        # 너무 작아지면 폴리곤 자체 제거
                        del self.polygons[poly_idx]
                    self.update()
                    return

        # current_polygon 쪽 점 삭제 (진행 중인 폴리곤에 대해서도)
        for i, (px, py) in enumerate(self.current_polygon):
            dx = x - px
            dy = y - py
            if dx * dx + dy * dy <= thr2:
                del self.current_polygon[i]
                self.update()
                return

    def undo_seg(self):
        # 진행 중인 폴리곤이면 마지막 점 삭제
        if self.current_polygon:
            self.current_polygon.pop()
            self.seg_preview_point = None
        # 아니면 마지막 확정 폴리곤 삭제
        elif self.polygons:
            self.polygons.pop()
        self.update()

    # ----------------------------
    #   좌표 변환
    # ----------------------------
    def _map_to_image(self, pos):
        if self.image is None or self.scale == 0:
            return None, None
        x = (pos.x() - self.offset_x) / self.scale
        y = (pos.y() - self.offset_y) / self.scale
        if x < 0 or y < 0 or x > self.img_w or y > self.img_h:
            return None, None
        return x, y


# ==========================
# YOLO 포맷 변환 함수
# ==========================
def yolo_from_boxes(boxes, img_w, img_h):
    """
    boxes: [(cls, x1,y1,x2,y2), ...] in pixel
    → "cls xc yc w h" (정규화) 줄들을 합친 문자열
    """
    lines = []
    for cls_id, x1, y1, x2, y2 in boxes:
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def yolo_from_polygons(polygons, img_w, img_h):
    """
    polygons: [(cls, [(x,y),...]), ...] in pixel
    → "cls x1 y1 x2 y2 ... xn yn" (정규화) 줄들을 합친 문자열
    """
    lines = []
    for cls_id, pts in polygons:
        coords = []
        for x, y in pts:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        lines.append(f"{cls_id} " + " ".join(coords))
    return "\n".join(lines)


# ==========================
#   메인 툴
# ==========================
class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO BBox + Seg Labeling Tool")

        self.image_dir = ""
        self.label_dir = ""
        self.image_files: list[str] = []
        self.current_index = -1
        self.last_enter_time = 0

        # 캐시
        self.label_cache_boxes: dict[str, list] = {}
        self.label_cache_polygons: dict[str, list] = {}
        self.modified_images: set[str] = set()

        self.mode = "bbox"

        self.init_ui()
        self.choose_mode_popup()

    # ----------------------------
    #   모드 선택 팝업
    # ----------------------------
    def choose_mode_popup(self):
        box = QMessageBox(self)
        box.setWindowTitle("라벨링 모드 선택")
        box.setText("어떤 라벨링 방식을 사용하시겠습니까?")
        bbox_button = box.addButton("Bounding Box", QMessageBox.ButtonRole.AcceptRole)
        seg_button = box.addButton("Segmentation", QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(bbox_button)
        box.exec()

        clicked = box.clickedButton()
        if clicked == seg_button:
            self.mode = "seg"
        else:
            self.mode = "bbox"

        self.image_label.mode = self.mode

    # ----------------------------
    #   UI 구성
    # ----------------------------
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 왼쪽 패널
        left_layout = QVBoxLayout()
        btn_img_dir = QPushButton("이미지 폴더 선택")
        btn_img_dir.clicked.connect(self.select_image_dir)
        btn_lbl_dir = QPushButton("라벨 폴더 선택")
        btn_lbl_dir.clicked.connect(self.select_label_dir)

        self.class_label = QLabel("현재 클래스: 0")

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_list_changed)

        self.clear_button = QPushButton("현재 이미지 라벨 삭제")
        self.clear_button.clicked.connect(self.clear_current_labels)

        self.save_button = QPushButton("전체 저장")
        self.save_button.clicked.connect(self.save_all)

        left_layout.addWidget(btn_img_dir)
        left_layout.addWidget(btn_lbl_dir)
        left_layout.addWidget(self.class_label)
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(self.clear_button)
        left_layout.addWidget(self.save_button)

        # 이미지 영역
        self.image_label = ImageLabel()
        self.image_label.setStyleSheet("background-color: black;")

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.image_label, 4)

        self.setCentralWidget(main_widget)
        self.resize(1500, 900)

    # ----------------------------
    #   키 입력
    # ----------------------------
    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        # 클래스 변경 (0~9)
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            cls_id = key - Qt.Key.Key_0
            self.image_label.set_current_class(cls_id)
            self.class_label.setText(f"현재 클래스: {cls_id}")

        # Ctrl+Z → Undo
        elif (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_Z:
            if self.mode == "bbox":
                self.image_label.undo_bbox()
            else:
                self.image_label.undo_seg()

        # Enter → 세그 폴리곤 닫기
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.mode == "seg":
                import time
                current_time = time.time()

                # ─────────────────────────
                # 1) 더블 엔터 감지 (0.25초 이내 두 번)
                # ─────────────────────────
                if current_time - self.last_enter_time < 0.25:
                    # 더블 엔터 → 폴리곤 닫기
                    self.image_label.finish_polygon()
                    self.last_enter_time = 0
                    return
                else:
                    self.last_enter_time = current_time

                # ─────────────────────────
                # 2) 일반 엔터 → 현재 마우스 위치에 점 추가
                # ─────────────────────────
                cursor_pos = self.image_label.mapFromGlobal(QCursor.pos())
                x, y = self.image_label._map_to_image(cursor_pos)

                if x is not None:
                    self.image_label.current_polygon.append((x, y))
                    self.image_label.seg_preview_point = None
                    self.image_label.update()




        # 좌우 화살표 → 이전/다음 이미지
        elif key == Qt.Key.Key_Right:
            self.next_image()
        elif key == Qt.Key.Key_Left:
            self.prev_image()
        else:
            super().keyPressEvent(event)

    # ----------------------------
    #   폴더 선택
    # ----------------------------
    def select_image_dir(self):
        d = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if d:
            self.image_dir = d
            self.load_image_list()

    def select_label_dir(self):
        d = QFileDialog.getExistingDirectory(self, "라벨 폴더 선택")
        if d:
            self.label_dir = d
            if self.image_files and self.current_index != -1:
                self.load_current_image(force_reload=True)

    # ----------------------------
    #   이미지 리스트 로드
    # ----------------------------
    def load_image_list(self):
        if not self.image_dir:
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = [
            f for f in sorted(os.listdir(self.image_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]

        self.list_widget.clear()
        self.list_widget.addItems(self.image_files)

        self.label_cache_boxes.clear()
        self.label_cache_polygons.clear()
        self.modified_images.clear()

        if self.image_files:
            self.current_index = 0
            self.list_widget.setCurrentRow(0)
            self.load_current_image(force_reload=True)

    def on_list_changed(self, row: int):
        if row < 0 or row >= len(self.image_files):
            return

        # 현재 이미지 캐시 저장
        self.update_cache_for_current_image()

        self.current_index = row
        self.load_current_image()

    # ----------------------------
    #   현재 이미지/라벨 로드
    # ----------------------------
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

    def load_current_image(self, force_reload: bool = False):
        img_path = self.current_image_path()
        if not img_path:
            return
        pix = QPixmap(img_path)
        if pix.isNull():
            return

        self.image_label.set_image(pix)

        img_name = self.image_files[self.current_index]

        # 캐시 우선
        if not force_reload and (
            img_name in self.label_cache_boxes or img_name in self.label_cache_polygons
        ):
            self.image_label.set_boxes(self.label_cache_boxes.get(img_name, []))
            self.image_label.set_polygons(self.label_cache_polygons.get(img_name, []))
            return

        # 라벨 파일에서 로드
        boxes = []
        polygons = []

        lbl_path = self.current_label_path()
        if lbl_path and os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])

                    # bbox: cls xc yc w h
                    if len(parts) == 5:
                        xc, yc, w, h = map(float, parts[1:])
                        x1 = (xc - w / 2) * self.image_label.img_w
                        y1 = (yc - h / 2) * self.image_label.img_h
                        x2 = (xc + w / 2) * self.image_label.img_w
                        y2 = (yc + h / 2) * self.image_label.img_h
                        boxes.append((cls_id, x1, y1, x2, y2))
                    # seg: cls x1 y1 x2 y2 ... xn yn
                    elif len(parts) > 5 and len(parts[1:]) % 2 == 0:
                        coords = list(map(float, parts[1:]))
                        pts = []
                        for i in range(0, len(coords), 2):
                            x = coords[i] * self.image_label.img_w
                            y = coords[i + 1] * self.image_label.img_h
                            pts.append((x, y))
                        if len(pts) >= 3:
                            polygons.append((cls_id, pts))

        self.image_label.set_boxes(boxes)
        self.image_label.set_polygons(polygons)
        self.image_label.setFocus()   # 이미지가 바뀔 때마다 포커스를 Label로 지정

    # ----------------------------
    #   캐시/체크
    # ----------------------------
    def update_cache_for_current_image(self):
        if self.current_index < 0 or not self.image_files:
            return
        img_name = self.image_files[self.current_index]
        self.label_cache_boxes[img_name] = self.image_label.get_boxes()
        self.label_cache_polygons[img_name] = self.image_label.get_polygons()
        self.modified_images.add(img_name)

    # ----------------------------
    #   라벨 삭제
    # ----------------------------
    def clear_current_labels(self):
        if self.current_index < 0:
            return
        reply = QMessageBox.question(
            self,
            "라벨 삭제 확인",
            "현재 이미지의 모든 라벨을 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.image_label.set_boxes([])
        self.image_label.set_polygons([])
        self.update_cache_for_current_image()

    # ----------------------------
    #   저장
    # ----------------------------
    def save_all(self):
        if not self.label_dir:
            QMessageBox.warning(self, "경고", "라벨 폴더를 먼저 선택하세요.")
            return

        # 현재 화면도 캐시에 반영
        self.update_cache_for_current_image()

        os.makedirs(self.label_dir, exist_ok=True)

        saved = 0
        for img_name in self.modified_images:
            img_path = os.path.join(self.image_dir, img_name)
            pix = QPixmap(img_path)
            if pix.isNull():
                continue
            img_w = pix.width()
            img_h = pix.height()

            boxes = self.label_cache_boxes.get(img_name, [])
            polygons = self.label_cache_polygons.get(img_name, [])

            lines = []
            if boxes:
                lines.extend(yolo_from_boxes(boxes, img_w, img_h).splitlines())
            if polygons:
                lines.extend(yolo_from_polygons(polygons, img_w, img_h).splitlines())

            lbl_path = os.path.join(
                self.label_dir,
                os.path.splitext(img_name)[0] + ".txt"
            )
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            saved += 1

        QMessageBox.information(
            self,
            "저장 완료",
            f"{saved}개 이미지에 대해 라벨을 저장했습니다.",
        )

    # ----------------------------
    #   prev / next
    # ----------------------------
    def next_image(self):
        if not self.image_files:
            return
        if self.current_index < len(self.image_files) - 1:
            self.list_widget.setCurrentRow(self.current_index + 1)

    def prev_image(self):
        if not self.image_files:
            return
        if self.current_index > 0:
            self.list_widget.setCurrentRow(self.current_index - 1)

    # ----------------------------
    #   종료 확인
    # ----------------------------
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "프로그램 종료",
            "정말 종료하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


# ==========================
#   main
# ==========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LabelingTool()
    win.show()
    sys.exit(app.exec())
