clear all;close all;clc
load('BZ.mat');
[m,n,k]=size(BZ_tensor); % x vs y vs time data
for j=1:k
A=BZ_tensor(:,:,j);
pcolor(A), shading interp, pause(0.2)
end