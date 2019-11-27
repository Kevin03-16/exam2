clear all, close all
clc

%Part 1: Select data distribution
TrainNum=1000;
alpha=[0.1,0.2,0.3,0.4];
plotData=1;

mu(:,1)=[7;7;1];
mu(:,2)=[7;-7;1];
mu(:,3)=[-7;-7;1];
mu(:,4)=[-7;7;1];
Sigma(:,:,1)=[15,1,1;1,14,1;1,1,11];
Sigma(:,:,2)=[16,1,1;1,12,1;1,1,12];
Sigma(:,:,3)=[12,1,1;1,10,1;1,1,13];
Sigma(:,:,4)=[9,1,1;1,13,1;1,1,14];

figure(1)
[TrainData,TrainNc,TrainLabel]=randGMM(TrainNum,alpha,mu,Sigma);
for l=1:length(alpha)
    scatter3(TrainData(1,TrainLabel==l),TrainData(2,TrainLabel==l),TrainData(3,TrainLabel==l)),hold on
end
legend('Class 1','Class 2','Class 3','Class 4')

%Part 2:Determine and evaluate the theoratically optimal MAP classifier
TestNum=10000;
[TestData,NcTest,LabelTest]=randGMM(TestNum,alpha,mu,Sigma);
figure(2)
for i=1:length(alpha)
    Posterior(i,:)=alpha(i)*evalGaussian(TestData,mu(:,i),Sigma(:,:,i));
end
[~,indi]=max(Posterior,[],1);
TrueData=TestData(:,LabelTest==indi);
ErrorData=TestData(:,LabelTest~=indi);
ErrorNum=length(ErrorData);
Perror=ErrorNum/TestNum
for r=1:length(alpha)%row is decision
    True=TestData(:,LabelTest==r&indi==r);
    scatter3(True(1,:),True(2,:),True(3,:)),hold on        
end
scatter3(ErrorData(1,:),ErrorData(2,:),ErrorData(3,:),'bx')
legend('Class 1 True','Class 2 True','Class 3 True','Class 4 True','Wrong classified')       

%part 3 neural network

%generate three separate training datasets from the data distruibution
for n=2:4
    N=10^n;
    [x,Nc,Label]=randGMM(N,alpha,mu,Sigma);
    [r,c]=size(x);
    PerceptronNum=10;
    target=zeros(4,length(Label));   
    %True label
    for i=1:length(Label)
        target(Label(i),i)=1; 
    end
    %train hidden layer size
    %allocate the space
    K=10;
    perf=zeros(1,K);
    CorrectProb=zeros(1,K);
    for hs=1:PerceptronNum
        for k = 1:K
            [xTrain,indTrain,xValidate,indValidate,Ntrain,Nvalidate] = Kfolder(k,K,N,x);    
            net=patternnet(hs,'traingdx');
            net.divideParam.trainRatio=1;
            net.divideParam.valRatio=0;
            net.divideParam.testRatio=0;
            net.trainParam.goal=0.001;
            net.trainParam.epochs=300;
            net.layers{1}.transferFcn='logsig';
            net.layers{2}.transferFcn='softmax';
            net.performFcn='crossentropy';
            net=train(net,xTrain,target(:,indTrain));
            y=net(xValidate);
            [~,i]=max(y);%predicted class;
            if find(i-Label(:,indValidate))
                perf(k)=length(find(i-Label(:,indValidate)));
            end
            CorrectProb(k)=(N-perf(k))/N;
        end
        P(hs)=mean(CorrectProb);
    end
    figure(n+1)
    plot(1:PerceptronNum,P(1:PerceptronNum))
    ylabel('correct Probability'),
    xlabel('hidden layer size')
    title(['In ', num2str(N),' dataset condition,performance under different hidden layer size'])
    [mle,Num]=max(P); 
    fprintf('the most appropriate number of perceptrons in the hidden layer in %d dataset is %d.\n',N,Num)
    
    %train the final neural network with appropriate model order using all
    %traiinf dataset
    net2=patternnet(Num,'traingdx');
    net2.divideParam.trainRatio=1;
    net2.divideParam.valRatio=0;
    net2.divideParam.testRatio=0;
    net2.trainParam.goal=0.001;
    net2.trainParam.epochs=300;
    net2.layers{1}.transferFcn='logsig';
    net2.layers{2}.transferFcn='softmax';
    net2.performFcn='crossentropy';
    net2=train(net2,x,target);
    yhat=net2(TestData);
    [~,l]=max(yhat);%predicted class;
    if find(l-LabelTest)
        perfTest=length(find(l-LabelTest));
    end
    CorrectProbTest(n-1)=(TestNum-perfTest)./TestNum;
    figure(n+4)
    CorrectData=TestData(:,l==LabelTest);
    ErrorData=TestData(:,l~=LabelTest);
    ErrorDataNum=length(find(l-LabelTest))
    scatter3(ErrorData(1,:),ErrorData(2,:),ErrorData(3,:),'rx'),hold on
    scatter3(CorrectData(1,:),CorrectData(2,:),CorrectData(3,:),'bo')
    legend('misclassified samples','classified samples')
    xlabel('x1')
    ylabel('x2')
    zlabel('x3')
    fprintf('Apply the neural network trained by %d dataset whose units in hiddden layer is %d to the test dataset, the probability of error estimate is %f.\n',N,Num,CorrectProbTest(n-1))
end

function [ xTrain,indTrain,xValidate,indValidate,Ntrain,Nvalidate ] = Kfolder(k,K,N,x)
dummy = ceil(linspace(0,N,K+1));
for m = 1:K
    indPartitionLimits(m,:) = [dummy(m)+1,dummy(m+1)];
end
indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
xValidate = x(:,indValidate); % Using folk k as validation set
if k == 1
    indTrain = [indPartitionLimits(k,2)+1:N];
elseif k == K
    indTrain = [1:indPartitionLimits(k,1)-1];
else
    indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
end
xTrain = x(:,indTrain); % using all other folds as training set
Ntrain = length(indTrain); Nvalidate = length(indValidate);
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

end

function [x,Nc,label]=randGMM(N,alpha,mu,Sigma)
d=size(mu,1);%dimensionality of samples
cum_alpha=[0,cumsum(alpha)];
u=rand(1,N);%randomly generate N numbers
x=zeros(d,N);
label=zeros(1,N);
for m=1:length(alpha)%iteration for the number of component    
    ind=find(cum_alpha(m)<u & u<=cum_alpha(m+1));
    x(:,ind)=randGaussian(length(ind),mu(:,m),Sigma(:,:,m));%generate samples for each component
    Nc(m)=length(ind);
    label(ind)=m;
end
end

%generate N samples from a Gaussian pdf with mean mu and covariance Sigma
function x=randGaussian(N,mu,Sigma)
n=length(mu);
z=randn(n,N);
A=Sigma^(1/2);
x=A*z+repmat(mu,1,N);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
%%%

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = (2*pi)^(-n/2) * det(Sigma)^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end





