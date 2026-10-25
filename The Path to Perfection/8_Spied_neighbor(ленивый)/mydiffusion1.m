function U = mydiffusion(D, S, B, U0, k, dt, t_end, display)
% MYDIFFUSION : movie of diffusion equation in 2 dimensions
%
% U = mydiffusion(D, S, B, U0, m, dt, t_end, movie);
%
% Solve the diffusion equation for u(t,x,y)
%
%     udot = D * laplacian(u) + S(x,y)
%
% on the 2-D unit square 0 <= x,y <= 1.
%
% Inputs:
%  D       diffusion coefficient, constant for all x, y, and t.
%  S(x,y)  source term, at (x,y) inside the unit square.
%  B(x,y)  boundary value u(t,x,y), at x=0 or x=1 or y=0 or y=1, at all times t.
%  U0(x,y) initial value u(0,x,y), at time 0.
%  k       number of interior discrete nodes in x direction or y direction.
%  dt      discrete time step.
%  t_end   solve the equation for t from 0 to t_end.
%  display what graphics to do: 
%          0           no graphics.
%          1 (default) plot u at each time step.
%          <string>    make a movie.
%
% Outputs:
%  U       k+2 by k+2 square array of final values of u(t_end,x,y).
%  File <display>.avi is a movie of the time evolution of u.
%
% John Gilbert, 6 March 2016, based on Frederic Gibou's Heat_2D.m


% Figure out whether to plot or make a movie file.
plotting = true;
movie = false;
if nargin == 8
    movie = ischar(display);
    plotting = movie || display;
end;

% Initialize the output/plotting grid and the first movie frame.
U = zeros(k+2, k+2);
for ix = 1:k+2
    for iy = 1:k+2
        if ix==1 || ix==k+2 || iy==1 || iy==k+2
            U(ix,iy) = B((ix-1)/(k+1), (iy-1)/(k+1));
        else
            U(ix,iy) = U0((ix-1)/(k+1), (iy-1)/(k+1));
        end;
    end;
end;
if plotting
    ax = [1 k+2 1 k+2 0 max(max(U))];
    surfc(U); axis(ax);
    title('Time 0');
    drawnow;
    fprintf('Hit space to begin simulation.\n');
    pause;
    if movie
        aviobj = avifile([display '.avi']);
        aviobj.Quality = 100;
        F = getframe(gca);
        aviobj = addframe(aviobj, F);
    end;
end;
    
% Discretize the unit square, with k interior points in each dimension.
m  = k*k;
dx = 1/(k+1);

% Discretize the initial value, source term, and boundary conditions
u = zeros(m,1);
s = zeros(m,1);
b = zeros(m,1);
for ix = 1:k
    for iy = 1:k
        i = iy + k*(ix-1);
        u(i) = U0(ix*dx, iy*dx);   % initial value
        s(i) =  S(ix*dx, iy*dx);   % source term
        if ix==1, b(i) = b(i) + B(0, iy*dx); end; % x=0 boundary
        if ix==k, b(i) = b(i) + B(1, iy*dx); end; % x=1 boundary
        if iy==1, b(i) = b(i) + B(ix*dx, 0); end; % y=0 boundary
        if iy==k, b(i) = b(i) + B(ix*dx, 1); end; % y=1 boundary
    end;
end;

% Assemble the matrix of the linear system.
A2 = laplacian2d_sparse2(k);
I  = speye(m);
A  = (dx^2)/(D*dt) * I + A2;

% Step time from 0 to t_end
for t = 0 : dt : t_end
    
    % Assemble the right-hand side
    r = (dx^2)/(D*dt) * u  +  (dx^2)/D * s  +  b;
    
    % Update u_n to u_n+1
    u = A \ r;
    
    % Draw the plot, update the movie
    if plotting
        U(2:k+1, 2:k+1) = reshape(u, k, k)';
        surfc(U); axis(ax);
        title(sprintf('Time %4.2f',t));
        drawnow;
    end;
    if movie
        F = getframe(gca);
        aviobj = addframe(aviobj, F);
    end;
    
end;

% Finalize the movie file
if movie
    aviobj = close(aviobj);
end;

% Map the final value of u back to the square grid.
U(2:k+1, 2:k+1) = reshape(u, k, k)';
