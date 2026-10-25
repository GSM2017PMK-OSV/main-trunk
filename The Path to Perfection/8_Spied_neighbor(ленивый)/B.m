function b = B(x,y)
% B : boundary values for diffusion equation
%
% b = B(x,y), for x=0 or x=1 or y=0 or y=1

% radiator in the middle of the south wall, else freezing

if y < .1 && x > .3 && x < .7
    b = 100;
else
    b = 0;
end;