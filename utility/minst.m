clear
clc
%% 首先将ubyte格式文件转化成常用的.mat文件格式

filename_images = 'C:\Users\81941\Desktop\MVRL-master\data/train-images.idx3-ubyte';
savename = 'C:\Users\81941\Desktop\MVRL-master\data/test_images'; 
FID = fopen(filename_images,'r');

MagicNumber=readint32(FID);
NumberofImages=readint32(FID);
rows=readint32(FID);
colums=readint32(FID);
savePath = [savename,'.mat'];
test_images = zeros(59992,rows*colums);
for i = 1:59992
            temp = fread(FID,(rows*colums), 'uchar');
            test_images(i,:) = temp';
end
save(savePath,'test_images')

filename_labels = 'C:\Users\81941\Desktop\MVRL-master\data/train-labels.idx1-ubyte';
savename = 'C:\Users\81941\Desktop\MVRL-master\data/test_labels'; 
FID = fopen(filename_labels,'r');

MagicNumber=readint32(FID);
NumberofImages=readint32(FID);
rows=readint32(FID);
colums=readint32(FID);
savePath = [savename,'.mat'];
test_labels = zeros(59992,1);
for i = 1:59992
            temp = fread(FID,(1), 'uchar');
            test_labels(i,:) = temp;
            
end
save(savePath,'test_labels')
 
