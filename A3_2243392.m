%Clear the command window
clear all;clc;

%Read all images from seperate folders such as Apple, Avocado, Lemon,
%Orange, Plum and Rasberry
%Add class to labels array
%For instance, Apple class is shown by 1, Avocado 2, Lemon 3, Orange 4,
%Plum 5 and Rasberry 6 my labels array will be labels = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6]

%for Apple folder
%folder is my folder path
folder='Database\Apple';
I=dir(fullfile(folder,'*.jpg'));
i = 0;
for k=1:5
  i = i + 1;
  filename=fullfile(folder,I(i).name);
  %read image from Apple folder
  I2{k}=imread(filename);
  %add my class to label array
  labels(k)=1;
  %take the green part of the rgb image 
  green_image = I2{k}(:,:,2); 
  %subtract the green part of the image from original image
  myImage{k} = I2{k} - green_image;  
end

%for Avocado folder
folder='Database\Avocado';
I=dir(fullfile(folder,'*.jpg'));
i = 0;
for k=6:10
  i = i +1;
  filename=fullfile(folder,I(i).name);
  %read image from Avocado folder
  I2{k}=imread(filename);
  %add my class to label array
  labels(k)=2;
  %take the green part of the rgb image 
  green_image = I2{k}(:,:,2); 
  %subtract the green part of the image from original image
  myImage{k} = I2{k} - green_image;  
end

%for Lemon folder
folder='Database\Lemon';
I=dir(fullfile(folder,'*.jpg'));
i = 0;
for k=11:15
  i = i +1;
  filename=fullfile(folder,I(i).name);
  %read image from Avocado folder
  I2{k}=imread(filename);
  %add my class to label array
  labels(k)=3;
  %take the green part of the rgb image 
  green_image = I2{k}(:,:,2); 
  %subtract the green part of the image from original image
  myImage{k} = I2{k} - green_image;
end

%for Orange folder
folder='Database\Orange';
I=dir(fullfile(folder,'*.jpg'));
i = 0;
for k=16:20
  i = i +1;
  filename=fullfile(folder,I(i).name);
  %read image from Avocado folder
  I2{k}=imread(filename);
  %add my class to label array
  labels(k)=4;
  %take the green part of the rgb image 
  green_image = I2{k}(:,:,2); 
  %subtract the green part of the image from original image
  myImage{k} = I2{k} - green_image;
end

%for Plum folder
folder='Database\Plum';
I=dir(fullfile(folder,'*.jpg'));
i = 0;
for k=21:25
  i = i +1;
  filename=fullfile(folder,I(i).name);
  %read image from Avocado folder
  I2{k}=imread(filename);
  %add my class to label array
  labels(k)=5;
  %take the green part of the rgb image 
  green_image = I2{k}(:,:,2); 
  %subtract the green part of the image from original image
  myImage{k} = I2{k} - green_image;
end

%for Rasberry folder
folder='Database\Rasberry';
I=dir(fullfile(folder,'*.jpg'));
i = 0;
for k=26:30
  i = i +1;
  filename=fullfile(folder,I(i).name);
  %read image from Avocado folder
  I2{k}=imread(filename);
  %add my class to label array
  labels(k)=6;
  %take the green part of the rgb image 
  green_image = I2{k}(:,:,2); 
  %subtract the green part of the image from original image
  myImage{k} = I2{k} - green_image;
end

%labels = ['Apple' 'Avocado0' 'Lemon' 'Orange' 'Plum' 'Rasberry'];
%labels = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6]

%I decided to use the color information for the feature extraction part,
%because I thought it would be a suitable feature extraction method
%when classifying since each fruit has a unique color.
for k = 1:30 
    %Split into RGB Channels
    Red = myImage{k}(:,:,1);
    Green = myImage{k}(:,:,2);
    Blue = myImage{k}(:,:,3);

    %Get histogram values for each channel
    [Red1, x] = imhist(Red);
    [Green1, x] = imhist(Green);
    [Blue1, x] = imhist(Blue);

    %Plot them together in one plot
    %figure;plot(x, Red1, 'Red', x, Green1, 'Green', x, Blue1, 'Blue');
    %Calculate the 'Mean' value for each channel
    mean_R=mean2(Red);
    mean_G=mean2(Green);
    mean_B=mean2(Blue);
    %sum all mean values
    summ = mean_B + mean_G + mean_R;
    %Calculate the 'Standard Deviation' value for each channel
    std_R=std2(Red);
    std_G=std2(Green);
    std_B=std2(Blue);
    %sum all standard deviation values
    summ2 = std_B + std_G + std_R;
    %add the color features to data array
    data(k,1)=(mean_R);
    data(k,2)=(std_R);
        
end

%I used Hold-out method to split my train and test data
%my division ratio is 0.3, so,we have 21 training data and 9 test data
%if division ratio is 0.2, my accuracy will be 100 and we have 24 training
%data and 6 test data
rng('default')
cv = cvpartition(size(data,1),'HoldOut',0.3);
test = cv.test;
k = 1;
m = 1;

%separate the training and test data
for i=1:30
    
    if test(i)==1
        test_data(k,1) = data(i,1);
        test_data(k,2) = data(i,2);
        test_labels(k) = labels(i);
        k = k+1;
    else
        train_data(m,1) = data(i,1);
        train_data(m,2) = data(i,2);
        train_labels(m) = labels(i);
        m = m+1;
    end
end

%I decided to use knn as a classification technique, I tried svm and naive bayes algorithms before,
%but I couldn't get a better result. Therefore, Knn has been a better classification algorithm for me.
%At the same time, I tried the k value separately as 1, 3, 5 and 7 in the knn algorithm,
%however, I got the best accuracy by looking at the 5 nearest neighbors.
%In addition, I tried different distance metrics, such as cityblock, hamming and euclidean
%but I got the best results in euclidean distance metric. 

%Create my model 
%k = [1,3,5,7]
%distance_metrics = ['cityblock','euclidean','hamming']
model = fitcknn(train_data,train_labels,'NumNeighbors',5,'Distance','euclidean');

%Classify the test data
predicted_labels = predict(model,test_data);

%Compute the accuracy of my model
x = classperf(test_labels,predicted_labels);
accuracy = x.CorrectRate;
accuracy = accuracy*100;
disp(['Accuracy: ', num2str(accuracy)]);





