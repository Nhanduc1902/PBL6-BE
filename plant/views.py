from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from rest_framework import status
from .models import Diseases, Predict
from .serializers import DiseasesSerializers, PredictSerializers
from django.views import View
from django.db.models import Q
from django.http import Http404
from django.conf import settings

from tensorflow.keras.models import load_model
import h5py
import cloudinary.uploader

import numpy as np
import cv2
import urllib
# Create your views here.
from .serializers import UploadedImageSerializer

class IndexClass(View):
    def get(self, request):
        return render(request,'plant/index.html')

class DiseasesList(APIView):
    def get(self, request, format=None):
        plant = Diseases.objects.all()
        mydata = DiseasesSerializers(plant, many=True)
        return Response(mydata.data)
    


class DiseasesDetail(APIView):
    """
    Retrieve, update or delete a snippet instance.
    """
    def get_object(self, pk):
        try:
            return Diseases.objects.get(pk=pk)
        except Diseases.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        plant = self.get_object(pk)
        mydata = DiseasesSerializers(plant)
        return Response(mydata.data)



class PredictList(APIView):
    # def get(self, request, format=None):
    #     #pre = Predict.objects.prefetch_related('diseases')
    #     pre = Predict.objects.all()
    #     #mydata = PredictSerializers(pre, many=True)
    #     #return Response(mydata.data)
    #     data = [f.to_json() for f in reversed(pre)]
    #     return Response(data)
    page_size = 10

    def get(self, request, format=None):
        # Lấy giá trị của tham số 'page' từ query parameters của URL
        page_number = request.GET.get("page", 1)
        queryset = Predict.objects.all()

        # Tạo một đối tượng Paginator từ django.core.paginator
        paginator = Paginator(queryset, self.page_size)

        try:
            # Lấy trang hiện tại
            page = paginator.page(page_number)
        except PageNotAnInteger:
            # Nếu 'page' không phải là một số nguyên, trả về trang đầu tiên
            page = paginator.page(1)
        except EmptyPage:
            # Nếu 'page' lớn hơn tổng số trang, trả về trang cuối cùng
            page = paginator.page(paginator.num_pages)

        # Serialize dữ liệu của trang hiện tại
        serializer = Predict(page, many=True)
        data = [f.to_json() for f in reversed(serializer)]
        # Trả về response phân trang
        return Response(data)


class PredictDetail(APIView):
    """
    Retrieve, update or delete a snippet instance.
    """
    def get_object(self, pk):
        try:
            return Predict.objects.get(pk=pk)
        except Predict.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        pre = self.get_object(pk)
        #mydata = PredictSerializers(pre)
        #return Response(mydata.data)
        return Response(pre.to_json())

class TestBase(APIView):
    def predict(self, image):
        classes = ['Pepper bell Bacterial_spot', 'Pepper bell healthy',
            'Potato Early blight' ,'Potato Late blight' ,'Potato healthy',
            'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
            'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
            'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot',
            'Tomato YellowLeaf Curl Virus', 'Tomato mosaic virus',
            'Tomato healthy']
        
        
        model = load_model('D:\PBL6\PBL6-main\my_model.h5')
        # model = model.fit()
        probabilities = model.predict(np.asarray([image]))[0]
        class_idx = np.argmax(probabilities)
        return [class_idx, probabilities[class_idx]]
        # return {classes[class_idx]: probabilities[class_idx]}

    def make_predict(self,filename):
        print ("make predict", filename)
        # img = cv2.imread(filename)

        try:
            req = urllib.request.urlopen(filename)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
            IMAGE_SHAPE = (256, 256)
            img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
            img = img /255
            prediction = self.predict(image=img)
            return prediction
            #res = {
            #    "class": list(prediction.keys())[0],
            #    "confident":list(prediction.values())[0]
            #}
        except Exception as e:
            print(f"Error in make_predict: {e}")
            return None
        

    def get(self, request, format=None):
        # rs = self.make_predict(filename='E:/123.jpg')
        
        rs = {"abcd":"edf"}
        return Response(rs, status=status.HTTP_201_CREATED)
    def post(self, request, format=None):
        
        upload_result = cloudinary.uploader.upload(request.data['link'])
        p = {}
        # p['Image'] = request.data['link']
        p['Image'] = upload_result['secure_url']
        pre = self.make_predict(filename=p['Image'])
        p['PredictResult'] = int(pre[0]) + 1
        p['Confident'] = pre[1]
        mydata = PredictSerializers(data = p)
        if mydata.is_valid():
            mydata.save()
            return Response(Predict.objects.all().last().to_json(), status=status.HTTP_201_CREATED)
        return Response("mydata.errors", status=status.HTTP_400_BAD_REQUEST)