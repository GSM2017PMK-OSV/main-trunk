function outvar = Heat(a,b,n,dt,t_final)
% HEAT: Time-dependent diffusion equation in 1D
% from Frederic Gibou's notes

aviobj = avifile('My_Movie.avi');
aviobj.Quality = 100;

x  = linspace(a,b,n);
dx = (b-a)/(n-1);
A  = zeros(n,n);
t  = 0; 
Un = cos(x); % initial temperature

while t < t_final % & count<...
    if t+dt > t_final
        dt = t_final-t;
    end
    t = t+dt;
    A(1,1) = 1; rhs(1) = cos(a)*exp(-t);
    A(n,n) = 1; rhs(n) = cos(b)*exp(-t);
    for i = 2:n-1
        A(i,i)   = 1 + 2*dt/dx/dx; 
        A(i,i-1) =      -dt/dx/dx; 
        A(i,i+1) =      -dt/dx/dx;
        rhs(i)   = Un(i); % no internal source of heat
    end
    Unp1 = A\rhs';
    Un = Unp1;
    
    exact = cos(x).*exp(-t);
    
    plot(x,exact,'r',x,Unp1,'bo'); axis([a b, 0 1]);
    legend('Exact','Numerical');
    xlabel('x');ylabel('Temperature');
    s = sprintf('Numerical Solution of the Temperature distribution at t =%4.2f',t);
    title(s);
    F = getframe(gca);
    aviobj = addframe(aviobj,F);
    pause(dt);
    
end
aviobj = close(aviobj);
outvar = Unp1;
end
