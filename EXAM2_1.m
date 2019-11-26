clear all, close all
clc

%Part 1: Select data distribution
n=3;TrainNum=1000;
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
    PerceptronNum=1;
    FinalMin=0;
    %Initialize the parameter

    t=zeros(4,length(Label));
    
    %True label
    for i=1:length(Label)
        t(Label(i),i)=1; 
    end
   
    while PerceptronNum<10
        %Initialize the parameter
        w1=randn(PerceptronNum,r);
        b1=zeros(PerceptronNum,1);
        w2=randn(4,PerceptronNum);
        b2=zeros(4,1);
        
        %10-Folder crossvalidation
        K=10;
        ValidateCorrect=zeros(1,K);
        for k = 1:K
            [xTrain,indTrain,xValidate,indValidate,Ntrain,Nvalidate] = Kfolder(k,K,N,x);    
            %update parameter
            iteration=1;
            Minimum=0;
            while iteration<101
                [Z1,A1,Z2,A2]=forward(xTrain,w1,b1,w2,b2);
                %MAP decision rul
                [~,Ind]=max(A2,[],1);
                %decision 
%                 if find(Ind-Label(indTrain))
%                     TrainError=length(find(Ind-Label(indTrain)));
%                 end

%                 MLE=-(1/Ntrain)*sum(C,2);
                c(iteration)=-sum(sum(t(:,indTrain).*log(A2),1),2);
                
                if iteration==1
                    Minimum=c(iteration);
                else
                    if c(iteration)<=Minimum
                        Minimum=c(iteration);
                        w1Output=w1;
                        b1Output=b1;
                        w2Output=w2;
                        b2Output=b2;
                    end
                end
%                 for i=1:Ntrain
%                     decision(Ind(i),i)=1;
%                     J=t(:,indTrain(i))-decision(:,i);
%                     if any(J)==0;
%                         correctA=correctA+1;
%                     end
%                 end     
                %back propogation
                [dZ1,dw1,db1,dZ2,dw2,db2] = backpropogation(A1,A2,w2,t(:,indTrain),Ntrain,xTrain);
                %gradient descend
                Rate=0.01;
                w1=w1-Rate*dw1;
                w2=w2-Rate*dw2;
                b1=b1-Rate*db1;
                b2=b2-Rate*db2;
                iteration=iteration+1;
                
            end
            w1=w1Output;
            w2=w2Output;
            b1=b1Output;
            b2=b2Output;
            
            
            %Validation            
            [~,~,~,yhat]=forward(xValidate,w1,b1,w2,b2);
            [~,Indv]=max(yhat,[],1);
%             if find(Indv-Label(indValidate))
%                 ErrorNum=length(find(Indv-Label(indValidate)));
%             end
%             for i=1:Nvalidate
%                 decisionValidate(Indv(i),i)=1;
%                 Jvalidate=t(:,indValidate(i))-decisionValidate(:,i);
%                 if any(Jvalidate)==0
%                     vc=vc+1;   
%                 end
%             end
            ValidationC(k)=-sum(sum(t(:,indValidate).*log(yhat),1),2);
%             ProbabilityCorrect(k)=(Nvalidate-ErrorNum)/Nvalidate;
        end
        MeanValidationC(PerceptronNum)=mean(ValidationC);
%         MeanCorrectProb(PerceptronNum)=mean(ProbabilityCorrect)
%         MeanProbabilityCorrect(PerceptronNum)=mean(ProbabilityCorrect);
        if PerceptronNum==1
            FinalMin=MeanValidationC(PerceptronNum);
        else
            if MeanValidationC(PerceptronNum)<FinalMin
                FinalMin=MeanValidationC(PerceptronNum);
                w1Final=w1;
                b1Final=b1;
                w2Final=w2;
                b2Final=b2;
                Num=PerceptronNum;
            end
                
        end
        PerceptronNum=PerceptronNum+1;
    end

    [~,~,~,predict]=forward(x,w1Final,b1Final,w2Final,b2Final);
    [~,i]=max(predict,[],1);
    if find(i-Label)
        k=length(find(i-Label));
    end
    Prob(n-1)=(N-k)/N;
    fprintf('when the number of dataset is %d, the number of units in hiddden layer is %d, the probability of correct decision is %f.\n',N,Num,Prob(n-1))
    figure(n+1)
    CorrectData=x(:,i==Label);
    ErrorData=x(:,i~=Label);
    scatter3(ErrorData(1,:),ErrorData(2,:),ErrorData(3,:),'rx'),hold on
    scatter3(CorrectData(1,:),CorrectData(2,:),CorrectData(3,:),'bo')
    legend('ErrorData','CorrectData')
    title(['the probability of correct decision for dataset with %d hidden layer units is %f',N,Num,Prob(n-1)])

    %%apply the trained neural network classifiers to the test dataset
    [~,~,~,TestResult]=forward(TestData,w1Final,b1Final,w2Final,b2Final);
    [~,MaxIndx]=max(TestResult,[],1);
    if find(MaxIndx-LabelTest)
        KtestError=length(find(MaxIndx-LabelTest));
    end
    ErrorProb(n-1)=KtestError/TestNum;
    fprintf('Apply the neural network trained by %d dataset whose units in hiddden layer is %d to the test dataset, the probability of error estimate is %f.\n',N,Num,ErrorProb(n-1))
    
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

function [dZ1,dw1,db1,dZ2,dw2,db2] = backpropogation(A1,A2,w2,label,N,InputData)
dZ2=A2-label;
db2=(1/N)*sum(dZ2,2);
dw2=(1/N)*dZ2*A1';
dZ1=w2'*dZ2.*A1.*(1-A1);
dw1=(1/N)*dZ1*InputData';
db1=(1/N)*sum(dZ1,2);
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
end

function [Z1,A1,Z2,A2]=forward(x,w1,b1,w2,b2)
Z1=w1*x+b1;%Z1 is [Perceptron x N]
A1=1./(1+exp(-Z1));
Z2=w2*A1+b2;%Z2 is [4 x N]
A2=exp(Z2)./sum(exp(Z2),1);
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





