from rest_framework import serializers
from .models import Diseases, Predict
from .models import UploadedImage


class DiseasesSerializers(serializers.ModelSerializer):
    NameDisease_VN = serializers.CharField(max_length=100)
    NameDisease_ENG = serializers.CharField(max_length=100)
    Symptom = serializers.CharField(max_length=2000)
    Pathogens = serializers.CharField(max_length=2000)
    Treatment = serializers.CharField(max_length=2000)

    class Meta:
        model = Diseases
        #fields = ('NameDisease_VN', 'NameDisease_ENG', 'Symptom','Pathogens','Treatment')
        fields = '__all__'


class PredictSerializers(serializers.ModelSerializer):
    #diseases = Diseases.objects.get(pk=1)
    #diseases = serializers.SlugRelatedField(
    #    many=True,
    #    read_only=True,
    #    slug_field='title'
    #)
    class Meta:
        model = Predict
        fields = '__all__'


class UploadedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedImage
        fields = ('id', 'image')