from ultralytics import YOLO
import inspect
print('YOLO.predict signature:')
print(inspect.signature(YOLO.predict))
print('\n--- Doc ---')
print(YOLO.predict.__doc__[:2000])
