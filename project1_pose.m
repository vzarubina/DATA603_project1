%% viewing images from pose.mat

clear all
close all
data = load('pose.mat');
pose = data.pose;
figure;
colormap gray
for j=1:13 
    subplot(3,5,j);
    imagesc(pose(:,:,j,1)); %the four dimensions: (img,img,pose,person)
end

%%
%Reshape data for PCA
n = 5;
dim = 48*40;
num_poses = 13;
poses1 = zeros(n*num_poses,dim);

for i = 1:n
    for j = 1:13
        aux = pose(:,:,j,i);
        poses1((i-1)+j,:) = aux(:)';
    end
end
%% Find means and center data
m1 = mean(poses1, 1);
Xc = poses1 - ones(n*num_poses,1)*m1;
%pca
nPCA = 40;
[U,Sig,V] = svd(Xc', 'econ');

Y = Xc*U(:,1:nPCA);



%% split test and train set
%want to use first 10 images for training and last 3 for testing
train = zeros(50,40);
test = zeros(15,40);
Y_test=Y;

train1 = Y(1:10,:);
train2 = Y(14:23,:);
train3 = Y(27:36,:);
train4 = Y(40:49,:);
train5 = Y(53:62,:);
train = vertcat(train1,train2,train3,train4,train5);

test1 = Y(11:13,:);
test2 = Y(24:26,:);
test3 = Y(37:39,:);
test4 = Y(50:52,:);
test5 = Y(63:65,:);
train = vertcat(test1,test2,test3,test4,test5);

%% Apply Bayesian decision theory
Y1=train;
Y2=test;
mu1 = mean(Y1,1);
mu2 = mean(Y2,1);
fprintf('norm(mu1-mu2) = %d\n',norm(mu1-mu2));

%% Find means and center data for PCA for KNN
m1 = mean(poses1, 1);
Xc = poses1 - ones(n*num_poses,1)*m1;
%pca
nPCA = 5;
[U,Sig,V] = svd(Xc', 'econ');

Y = Xc*U(:,1:nPCA);
%% split test and train set
%want to use first 10 images for training and last 3 for testing
train = zeros(50,40);
test = zeros(15,40);
Y_test=Y;

train1 = Y(1:10,:);
train2 = Y(14:23,:);
train3 = Y(27:36,:);
train4 = Y(40:49,:);
train5 = Y(53:62,:);
train = vertcat(train1,train2,train3,train4,train5);

test1 = Y(11:13,:);
test2 = Y(24:26,:);
test3 = Y(37:39,:);
test4 = Y(50:52,:);
test5 = Y(63:65,:);
train = vertcat(test1,test2,test3,test4,test5);
%% Gaussian KNN
k = 20;


% test 
errors1 = 0; 
for i = 1: length(test1)
    % fidn the distance between the training data and the current test
    % candidate
    x1_diff = abs(train1 - test1(i));
    x2_diff = abs(train2 - test1(i));
    % sort the distance vectors to see the nearest points
    x1_diff = sort(x1_diff);
    x2_diff = sort(x2_diff);
    % find the classes of hte first k points
    x_diff = x1_diff - x2_diff;
    x_diff = sign(x_diff);
    x_diff = x_diff(1:k);
    % classify each point using the first k points
    if sum(x_diff > 0) 
        errors1 = errors1 + 1;
    end
end

%% test test2
errors2 = 0; 
for i = 1: length(test2)
    x1_diff = abs(train2 - test2(i));
    x2_diff = abs(train3 - test2(i));
    x1_diff = sort(x1_diff);
    x2_diff = sort(x2_diff);
    x_diff = x1_diff - x2_diff;
    x_diff = sign(x_diff);
    x_diff = x_diff(1:k);
    if sum(x_diff < 0) 
        errors2 = errors2 + 1;
    end
end

%% test test 3
errors3 = 0; 
for i = 1: length(test3)
    x1_diff = abs(train3 - test3(i));
    x2_diff = abs(train4 - test3(i));
    x1_diff = sort(x1_diff);
    x2_diff = sort(x2_diff);
    x_diff = x1_diff - x2_diff;
    x_diff = sign(x_diff);
    x_diff = x_diff(1:k);
    if sum(x_diff < 0) 
        errors2 = errors2 + 1;
    end
end

%% test test4
errors4 = 0; 
for i = 1: length(test4)
    x1_diff = abs(train4 - test4(i));
    x2_diff = abs(train5 - test4(i));
    x1_diff = sort(x1_diff);
    x2_diff = sort(x2_diff);
    x_diff = x1_diff - x2_diff;
    x_diff = sign(x_diff);
    x_diff = x_diff(1:k);
    if sum(x_diff < 0) 
        errors2 = errors2 + 1;
    end
end

%% test 5
errors5 = 0; 
for i = 1: length(test5)
    x1_diff = abs(train5 - test5(i));
    x2_diff = abs(train1 - test5(i));
    x1_diff = sort(x1_diff);
    x2_diff = sort(x2_diff);
    x_diff = x1_diff - x2_diff;
    x_diff = sign(x_diff);
    x_diff = x_diff(1:k);
    if sum(x_diff < 0) 
        errors2 = errors2 + 1;
    end
end

%find the total number of errors by summing the test error from each class
errors1
errors2
errors3
errors4
errors5
errors = errors1 + errors2 +errors3 + errors4 +errors5
