
%viewing face images from data.mat

data = load('data.mat');
face = data.face;
face_neutral = face(:,:,1:3:end);
face_exp = face(:,:,2:3:end);
face_illum = face(:,:,3:3:end);


figure;
colormap gray
for j=1:20 
    subplot(4,5,j);
    imagesc(face_neutral(:,:,j));
end

figure;
colormap gray
for j=1:20 
    subplot(4,5,j);
    imagesc(face_exp(:,:,j));
end

figure;
colormap gray
for j=1:20 
    subplot(4,5,j);
    imagesc(face_illum(:,:,j));
end

%% viewing images from illumination.mat

data = load('illumination.mat');
illum = data.illum;
% display the first subject
figure;
colormap gray
i=1;
for j=1:21 
    subplot(3,7,j);
    imagesc(reshape(illum(:,j,i),[48,40]));
end

%% pca
[d1,d2,n] = size(face_neutral);
X1=zeros(n,d1*d2);
X2=zeros(n,d1*d2);

for j=1:n
    aux = face_neutral(:,:,j);
    X1(j,:) = aux(:)'; 
    aux = face_exp(:,:,j);
    X2(j,:) = aux(:)';
end
n1=n; %200
n2=n;
m1=mean(X1,1); 
m2=mean(X2,1); 
Xc1=X1-ones(n1,1)*m1;
S1=Xc1'*Xc1;
Xc2=X2-ones(n2,1)*m2;
S2=Xc2'*Xc2;
Sw=S1+S2; 
Sb=(m1-m2)'*(m1-m2);
[W,lam0]=eig(Sb,Sw);
lam=diag(lam0);
[lam,isort]=sort(lam,'descend');
W=W(:,isort);
w=W(:,1);
X=[X1;X2];
D1=1:n1;
D2=n1+1:n1+n2;


[U,sig,V] = svd(X','econ');
nPCA = 20;
%project to nPCA-dimensional space
Y = X*U(:,1:nPCA);
figure;
hold on;
grid;
plot3(Y(D1,1),Y(D1,2),Y(D1,3),'.','Markersize',20,'color','k');
plot3(Y(D2,1),Y(D2,2),Y(D2,3),'.','Markersize',20,'color','r');
%look at it in 3-d
view(3);

%% split training and test sets
i=100;
D1_train = D1(:,1:i);
D1_test = D1(:,i+1:200);
%% Apply Bayesian decision theory
Y1=Y(D1_train,:);
Y2=Y(D1_test,:);
mu1 = mean(Y1,1);
mu2 = mean(Y2,1);
fprintf('norm(mu1-mu2) = %d\n',norm(mu1-mu2));

%% estimate covariance matrices

%first center the data
Y1c=Y1-ones(i,1)*mu1;
Y2c=Y2-ones(i,1)*mu2;
S1=Y1c'*Y1c/i;
S2=Y2c'*Y2c/i;
figure;
imagesc(S1);
colorbar;
figure;
imagesc(S2);
colorbar;


%% define discriminant function
iS1=inv(S1);
iS2=inv(S2);
mu1=mu1';
mu2=mu2';
w0= 0.5*(log(det(S2)/det(S1))) - 0.5*(mu1'*iS1*mu1-mu2'*iS2*mu2);
g=@(x) - 0.5*x'*(iS1-iS2)*x + x'*(iS1*mu1-iS2*mu2)+w0;

%classify the illuminated faces
Y3=zeros(n,nPCA);
label=zeros(n,1);
for j=1:n
    aux=face_illum(:,:,j);
    y=(aux(:)'*U(:,1:nPCA))';
    Y3(j,:) = y';
    label(j)=sign(g(y));    
end
iplus=find(label>0);
iminus=find(label<0);
fprintf('#iplus = %d,#iminus = %d\n',length(iplus),length(iminus));



%% Using k-nn 
%preprocess using PCA w/ smaller dimensional space
[d1,d2,n] = size(face_neutral);
X1=zeros(n,d1*d2);
X2=zeros(n,d1*d2);
for j=1:n
    aux = face_neutral(:,:,j);
    X1(j,:) = aux(:)';
    aux = face_exp(:,:,j);
    X2(j,:) = aux(:)';
end
n1=n;
n2=n;
m1=mean(X1,1);
m2=mean(X2,1);
Xc1=X1-ones(n1,1)*m1;
S1=Xc1'*Xc1;
Xc2=X2-ones(n2,1)*m2;
S2=Xc2'*Xc2;
Sw=S1+S2;
Sb=(m1-m2)'*(m1-m2);
[W,lam0]=eig(Sb,Sw);
lam=diag(lam0);
[lam,isort]=sort(lam,'descend');
W=W(:,isort);
w=W(:,1);
X=[X1;X2];
D1=1:n1;
D2=n1+1:n1+n2;


[U,sig,V] = svd(X','econ');
nPCA = 5;
%project to nPCA-dimensional space
Y = X*U(:,1:nPCA)
figure;
hold on;
grid;
plot3(Y(D1,1),Y(D1,2),Y(D1,3),'.','Markersize',20,'color','k');
plot3(Y(D2,1),Y(D2,2),Y(D2,3),'.','Markersize',20,'color','r');
%look at it in 3-d
view(3);


%%
% Gaussian KNN
x1 = D1;
x2 = D2;

k = 20;
% Break up the data into training and testing sets 
i=150;
x1_train = x1(1:i);
x2_train = x2(1:i);
x1_test = x1(i+1:200);
x2_test = x2(i+1:200);

% test x1_test
errors1 = 0; 
for i = 1: length(x1_test)
    % fidn the distance between the training data and the current test
    % candidate
    x1_diff = abs(x1_train - x1_test(i));
    x2_diff = abs(x2_train - x1_test(i));
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

% test x2_test
errors2 = 0; 
for i = 1: length(x2_test)
    % fidn the distance between the training data and the current test
    % candidate
    x1_diff = abs(x1_train - x2_test(i));
    x2_diff = abs(x2_train - x2_test(i));
    % sort the distance vectors to see the nearest points
    x1_diff = sort(x1_diff);
    x2_diff = sort(x2_diff);
    % find the classes of the first k points
    x_diff = x1_diff - x2_diff;
    x_diff = sign(x_diff);
    x_diff = x_diff(1:k);
    % classify each point using the first k points
    if sum(x_diff < 0) 
        errors2 = errors2 + 1;
    end
end

%find the total number of errors by summing the test error from each class
errors1
errors2
errors = errors1 + errors2
