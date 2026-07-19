from ultralytics import YOLO
model = YOLO('/home/sagnik/Projects/docker-composes/manga-library/unified-workers/models_backup/models--ShadowB--Manga109-panel-balloon-text-yolov26-segmentation/snapshots/3a860269ee0beb43ce9f31d82c7851441eb178ae/best.pt')
# We can't easily see the export code without looking at the source, but we can look at the source!
import inspect
import ultralytics.engine.exporter as exporter
print("Exporter source:", inspect.getsource(exporter.Exporter.__init__))
