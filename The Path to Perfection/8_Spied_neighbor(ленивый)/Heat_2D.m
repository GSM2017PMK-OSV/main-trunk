function outvar = Heat_2D(D,S,BC,Initial_Data,a,b,c,d,m,n,dt,t_final)

aviobj = avifile('My_Movie.avi');
aviobj.Quality = 100;
% Discretize the domain:
x = linspace(a,b,m);
y = linspace(c,d,n);
dx=(b-a)/(m-1);
dy=(d-c)/(n-1);
mn=m*n;
% Initialize the variables:
A=sparse(mn,mn);
t=0;
% Initial condition:
for i=1:m
    for j=1:n
        r=(j-1)*m+i;
        Un(r)=Initial_Data(x(i),y(j),t);
    end
end
% Time stepping:
while t<t_final % & count<...
    if t+dt>t_final
        dt=t_final-t;
    end
    t=t+dt;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Assemble the linear system: %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Treat the interior points:
    for i=2:m-1
        for j=2:n-1
            r=(j-1)*m+i;
            A(r,r )=1 + 2*dt*D/dx/dx + 2*dt*D/dy/dy;
            A(r,r-1)= -dt*D/dx/dx;
            A(r,r+1)= -dt*D/dx/dx;
            A(r,r-m)= -dt*D/dy/dy;
            A(r,r+m)= -dt*D/dy/dy;
            rhs(r)=Un(r)+dt*S(x(i),y(j),t);
        end
    end
    % Treat the boundary conditions:
    for j=1:n
        i=1; % Left boundary condition
        r=(j-1)*m+i;
        A(r,r)=1; rhs(r)=BC(a,y(j),t);
        i=m; % Right boundary condition
        r=(j-1)*m+i;
        A(r,r)=1; rhs(r)=BC(b,y(j),t);
    end
    for i=1:m
        j=1; % Bottom boundary condition
        r=(j-1)*m+i;
        A(r,r)=1; rhs(r)=BC(x(i),c,t);
        j=n; % Top boundary condition
        r=(j-1)*m+i;
        A(r,r)=1; rhs(r)=BC(x(i),d,t);
    end
    % Solve the linear system:
    Unp1=A\rhs';
    
    Un=Unp1;
    % Need to put Un, which is now a vector, into a grid form.
    for i=1:m
        for j=1:n
            r=(j-1)*m+i;
            U_grid(i,j)=Un(r);
        end
    end
    mesh(x,y,U_grid); axis([a b, c d, 0 2]);
    xlabel('x');ylabel('y'); zlabel('Temperature');
    s=sprintf('Numerical Solution of the Temperature distribution at t =%4.2f',t);
    title(s);
    F=getframe(gca);
    aviobj=addframe(aviobj,F);
    pause(dt/10);
end
aviobj=close(aviobj);
outvar=Un;
end

