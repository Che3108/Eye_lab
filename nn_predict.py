#!/usr/bin/python3

# отключение логирования
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Подключение GPU
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Библиотеки
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Lambda  # Стандартные слои Keras
from tensorflow.keras.layers import ZeroPadding2D, Add, UpSampling2D           # Стандартные слои Keras
from tensorflow.keras.layers import LeakyReLU, Concatenate
from tensorflow.keras.layers import Dropout # Стандартные слои Keras
from tensorflow.keras.regularizers import l2 # Регуляризатор l2
from tensorflow.keras.optimizers import Adam # Оптимизатор Adam
from tensorflow.keras.models import Model # Абстрактный класс Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw, ImageFont # Модули работы с изображениями
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb # Модули конвертации между RGB и HSV
import tensorflow.keras.backend as K # бэкенд Keras
import math # Импортируем модуль math
import pandas as pd # Пандас для работы с csv
import numpy as np # numpy массивы
import matplotlib.pyplot as plt # графики

class YOLOv3Predict(object):
    def __init__(self, name_classes, use_weights_flag = False, 
                 weights_path = 'yolo.h5', ignore_thresh = 0.1):
        self.name_classes = name_classes
        self.num_classes = len(name_classes)
        self.input_shape = (416, 416)
        self.num_conv = 0

        # Порог вероятности обнаружения объекта
        self.ignore_thresh = ignore_thresh

        # Сетки трех уровней для расчета координат
        self.grids = []
        for i in [13, 26, 52]:
            l = [[j] for j in list(range(i))]
            for_concat = (np.array([l]*len(l)),
                          np.array([[j]*len(l) for j in l]))
            grid = np.concatenate(for_concat, -1)
            grid = np.expand_dims(grid, 2)
            self.grids.append(grid)

        # анкоры
        self.anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                                 [[ 30, 61], [ 62,  45], [ 59, 119]],
                                 [[ 10, 13], [ 16,  30], [ 33,  23]]])
        self.anchors_2 = self.anchors.reshape((np.prod((self.anchors.shape[0], self.anchors.shape[1]),0), self.anchors.shape[-1]))
        self.num_anchors = len(self.anchors_2)
        self.num_sub_anchors = self.anchors.shape[1]
        self.num_layers = int(self.num_anchors/self.num_sub_anchors)

        # создаем модель и грузим веса
        self.model = self.create_YOLOv3()
        if use_weights_flag: self.model.load_weights(weights_path)
    
    def __ConvBL(self,inputs,*args,**kwargs):
        ''' Функция создания блока Conv2D, BatchNormalization, LeakyRelu
                Входные параметры:
                      inputs - Стартовый слой, к которому добавляется Res-блок
                      args - массив неименованных параметров
                      kwargs  - массив именованных параметров
         '''
        new_kwargs = {'use_bias': False} # создаем новый массив именованных параметров, добавляя параметр use_bias
        new_kwargs['kernel_regularizer'] = l2(5e-4) # добавляем параметр kernel_regularizerpadding
        new_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same' # добавляем параметр  padding
        new_kwargs.update(kwargs) # Добавляем уже имеющиеся в kwargs gfhfvtnhs
        x = Conv2D(*args, **new_kwargs) (inputs) # Добавляем Conv2D слой
        x = BatchNormalization() (x) # Добавляем слой BatchNormalization
        #x = Dropout(0.2) (x)
        x = LeakyReLU(alpha=0.1) (x) # Добавляем слой LeakyRelu
        #self.num_conv += 1
        #print(f'### ConvBL_{self.num_conv} ###')
        return x
    
    def __resblock(self,inputs,num_filters,num_blocks):
        ''' Функция создания Residual блока.
                Входные параметры:
                      inputs - Стартовый слой, к которому добавляется Res-блок
                      num_filters - количество нейронов
                      num_blocks  - количество блоков 
         '''
        x = ZeroPadding2D(((1, 0), (1, 0)))(inputs) # Увеличиваем размерность на один шаг влево и вверх
        x = self.__ConvBL(x, num_filters, (3, 3), strides=(2, 2)) # Добавляем блок ConvBL
        for i in range(num_blocks): # Пробегаем в цикле num_blocks-раз (суммируя слои с помощью Add())
            y = self.__ConvBL(x, num_filters // 2, (1, 1))
            y = self.__ConvBL(y, num_filters, (3, 3))
            x = Add() ([x, y])
        return x # Возвращаем слой

    def __sigmoid(self, x): # На вход подаем массив данных
        return 1/(1+np.exp(-x)) # Возвращаем сигмоиду для всех элементов массива

    def load_image(self, input_image):
        # если указан путь
        if isinstance(input_image, str):
            # размер исходного изображения
            with Image.open(input_image) as img:
                size_img = img.size
            image_arr = image.load_img(input_image, target_size = self.input_shape)
            image_arr = np.array(image_arr)
            image_arr = image_arr[np.newaxis,...]
            return size_img, image_arr
        elif isinstance(input_image, np.ndarray):
            return input_image.shape[:-1], input_image[np.newaxis,...]
        # необходимо прикрутить декорирование
            

    def create_YOLOv3(self):
        '''Функция создания модели YOLOv3
                Входные параметры:
                      inputs - Входной слой модели
                      num_sub_anchors - количество анкоров в каждом уровне сеток
        '''
        #--------------------
        # Базовая часть модели YOLOv3
        #--------------------

        # Состоит из Conv2D-слоев и Residual-блоков. Residual-блок - это блок использующий информацию из предыдущих слоев.
        # С помощью слоя Add (Суммируется текущий слой и один из предыдущих), что позволяет избежать проблему потери информации
        # Количество resedual блоков и архитектура сети взята из документации YOLOv3
        inputs = Input(self.input_shape + (3,))
        x = self.__ConvBL (inputs, 32, (3, 3)) # Добавляем каскад из трех слоев (Conv2D, BatchNormalization и Leaky)
        x = self.__resblock (x, 64, 1) # Добавляем 1 resedual-блок с 64 нейронами
        x = self.__resblock (x, 128, 2) # Добавляем 2 resedual-блока с 128 нейронами
        x = self.__resblock (x, 256, 8) # Добавляем 8 resedual-блоков с 256 нейронами
        x = self.__resblock (x, 512, 8) # Добавляем 8 resedual-блоков с 512 нейронами
        x = self.__resblock (x, 1024, 4) # Добавляем 4 resedual-блоков с 1024 нейронами
        base_model = Model(inputs, x) # Создаем базовую часть модели YOLOv3

        #--------------------
        # Detection часть модели YOLOv3
        #--------------------

        # Выделяем три выхода сети, соответсвующих различным уровням сетки
        # 13 x 13 (обнаружение больших объектов)
        x = self.__ConvBL(base_model.output, 512, (1, 1))
        x = self.__ConvBL(x, 1024, (3, 3))
        x = self.__ConvBL(x, 512, (1, 1))
        x = self.__ConvBL(x, 1024, (3, 3))
        x = self.__ConvBL(x, 512, (1, 1))
        # Выделяем первый выход модели, соответствующий размерности 13 х 13
        y1 = self.__ConvBL(x, 1024, (3,3))
        y1 = Conv2D(self.num_sub_anchors * (self.num_classes + 5), (1, 1), padding = 'same', kernel_regularizer = l2(5e-4)) (y1)

        # 26x26 (обнаружение средних объектов)
        # Размерность текущего выхода сети равна 13 х 13. Необходимо увеличить ее до 26 x 26 и
        # объеденить со 152-ым слоем (размерностью 26 x 26)
        x = self.__ConvBL(x, 256, ( 1, 1))
        x = UpSampling2D(2) (x) # Увеличиваем размерность до 26 на 26, использую UpSampling
        x = Concatenate()([x,base_model.layers[152].output])
        # Добавляем 5 блоков ConvBL
        x = self.__ConvBL(x, 256, (1, 1))
        x = self.__ConvBL(x, 512, (3, 3))
        x = self.__ConvBL(x, 256, (1, 1))
        x = self.__ConvBL(x, 512, (3, 3))
        x = self.__ConvBL(x, 256, (1, 1))
        # Выделяем второй выход модели, соответствующий размерности 26 х 26
        y2 = self.__ConvBL(x, 512, (3, 3))
        y2 = Conv2D(self.num_sub_anchors * (self.num_classes + 5), (1, 1), padding = 'same', kernel_regularizer = l2(5e-4)) (y2)

        # 52 x 52 (обнаружение маленьких объектов)
        # Размерность текущего выхода сети равна 26 х 26. Необходимо увеличить ее до 52 x 52 и
        # объеденить со 92-ым слоем (размерностью 52 x 52)
        x = self.__ConvBL(x, 128, ( 1, 1)) 
        x = UpSampling2D(2) (x)  # Увеличиваем размерность до 52 на 52, использую UpSampling
        x = Concatenate()([x,base_model.layers[92].output])
        # Добавляем 5 блоков ConvBL
        x = self.__ConvBL(x, 128, (1, 1))
        x = self.__ConvBL(x, 256, (3, 3))
        x = self.__ConvBL(x, 128, (1, 1))
        x = self.__ConvBL(x, 256, (3, 3))
        x = self.__ConvBL(x, 128, (1, 1))
        # Выделяем третий выход модели, соответствующий размерности 52 х 52
        y3 = self.__ConvBL(x, 256, (3, 3))
        y3 = Conv2D(self.num_sub_anchors * (self.num_classes + 5), (1, 1), padding = 'same', kernel_regularizer = l2(5e-4)) (y3)

        return Model(inputs, [y1, y2, y3]) # Возвращаем модель

    def prediction(self, input_image):
        real_img_size, img_arr = self.load_image(input_image)
        pred_model = self.model.predict(img_arr)
        temp_l = [[] for i in range(self.num_classes)]
        for n, v in enumerate(pred_model):
            grid = self.grids[n]
            pred = v.reshape((v.shape[1], v.shape[2], 
                              self.num_sub_anchors, 5 + self.num_classes))
            xy_param = pred[..., :2]
            box_xy = (self.__sigmoid(xy_param) + grid)/grid.shape[:2]
            pred[..., :2] = box_xy

            anchors_tensor = np.reshape(self.anchors[n], (1,1,1,self.num_sub_anchors,2))
            wh_param = pred[..., 2:4]
            box_wh = np.exp(wh_param) * anchors_tensor / self.input_shape
            pred[..., 2:4] = box_wh

            pred[:,:,:,4:] = self.__sigmoid(pred[:,:,:,4:])
            pred = pred[np.where(pred[:,:,:,4] > self.ignore_thresh)]
            if len(pred) != 0:
                for j in pred:
                    temp_l[np.argmax(j[5:])].append(j[:4].tolist())
        return temp_l

w = 'YOLOv3_auto_el1000__opt0.001__ep0_100_class_1.h5'
name_classes = ['Глаз']
model_yolo_v3 = YOLOv3Predict(name_classes, use_weights_flag = True, weights_path = w, ignore_thresh = 0.05)
#plot_model(model_yolo_v3.create_YOLOv3(), show_shapes=True)

def paint_box(img):
    
    #img[32:416+32,112:416+112,:]
    arg = model_yolo_v3.prediction(img)[0]
    #img = Image.fromarray(img)
    size = img.shape[0]
    #ramka = 2
    img_2 = np.full(img.shape, [127,127,127]).astype('uint8')
    for v in arg:
        #color = (255, 63, 63)
        bb_info = np.array(v)
        bb_info = np.floor(bb_info*size).astype(int)

        # Только глаза
        img_2[bb_info[1]-bb_info[3]//2:bb_info[1]-bb_info[3]//2+bb_info[3], bb_info[0]-bb_info[2]//2:bb_info[0]-bb_info[2]//2+bb_info[2], :] = img[bb_info[1]-bb_info[3]//2:bb_info[1]-bb_info[3]//2+bb_info[3], bb_info[0]-bb_info[2]//2:bb_info[0]-bb_info[2]//2+bb_info[2], :]
        
        # Глаза замазаны
        #img[bb_info[1]-bb_info[3]//2:bb_info[1]-bb_info[3]//2+bb_info[3], bb_info[0]-bb_info[2]//2:bb_info[0]-bb_info[2]//2+bb_info[2], :] = np.full((bb_info[3], bb_info[2], 3), [63,255,63])
        
        
        
        
        #print(bb_info)
        #border = Image.new('RGBA', (bb_info[2]+2*ramka, bb_info[3]+2*ramka), color+(255,))
        #plt.imshow(border)
        #box = Image.new('RGBA', (bb_info[2], bb_info[3]), color+(63,))
        #border.paste(box, (ramka, ramka))
        #img.paste(border, (bb_info[0]-bb_info[2]//2-1, bb_info[1]-bb_info[3]//2-1), mask=border)

    return img_2
