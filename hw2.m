clear all;close all;clc

% dt = 2;
% t = 1845:dt:1903;
hare = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 ...
     137 18 22 52 83 18 10 9 65];
lynx = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 ...
     26 18 37 50 35 12 12 25];
% 
% n = length(t);
% for j = 2:n-1
%     xdot(j-1) = (lynx(j+1)-lynx(j-1))/(2*dt);
% end
% x1s = x1(2:n-1);
% A=[x1s x1s.^2 x1s.^3 sin(x1s) cos(x1s)];
% xi1 = A\x1dot.';

x0 = [20 32];
tspan = 1:30;
%tspan = 1845:0.2:1903;

% [t,y1] = ode45('rhs_dyn1',t,x0,[]);
% 
% x1_pre = y1(:,1);
% x2_pre = y1(:,2);
% 
% figure(1)
% plot(t,hare,'r')
% hold on
% plot(t,x1_pre,'b')
% xlabel('time')
% ylabel('hare population')
% title('comparison of hare population using model include sine and cosine')
% saveas(gcf,'hare_trig.jpg')
% figure(2)
% plot(t,lynx,'r')
% hold on
% plot(t,x2_pre,'b')
% xlabel('time')
% ylabel('lynx population')
% title('comparison of lynx population using model include sine and cosine')
% saveas(gcf,'lynx_trig.jpg')

[t,y2] = ode45('rhs_dyn2',tspan,x0,[]);

x1_pre2 = y2(:,1);
x2_pre2 = y2(:,2);

figure(1)
plot(t,hare,'r')
hold on
plot(t,x1_pre2,'b')
xlabel('time')
ylabel('hare population')
title('comparison of hare population using model include polynomials')
saveas(gcf,'hare_poly.jpg')
figure(2)
plot(t,lynx,'r')
hold on
plot(t,x2_pre2,'b')
xlabel('time')
ylabel('lynx population')
title('comparison of lynx population using model include polynomials')
saveas(gcf,'lynx_poly.jpg')
