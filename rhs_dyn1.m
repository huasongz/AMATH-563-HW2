function rhs = rhs_dyn1(t,x,dummy)
rhs = [-2.0162477*cos(x(1))+4.12257185*sin(x(1))*cos(x(2))+1.33460823*cos(x(1))*sin(x(2));
    -0.99132507*sin(x(1))-1.04557915*cos(x(1))+0.22519842*sin(x(2))+0.53401031*sin(x(1))*cos(x(2))];
end
