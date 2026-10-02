clc;clear all;
% initial conditions of the different periodic orbots
ic_a = [ 1.2, 0, 0, -1.0493575 ];
ic_b = [ 0.994, 0, 0, -2.0317326295573368357302057924 ];
ic_c = [  0.994, 0, 0, -2.00158510637908252240537862224 ];

t_range_a = [ 0, 6.2 ];
t_range_b = [ 0, 11.2 ];
t_range_c = [ 0, 17.1 ];
t_range_L = [ 0, 10 ];

m2 = 5.97219 * 10^24; % mass of earth
m1 = 7.34767309 *10^22; % mass of the moon
mu = m1 / ( m1 + m2 );
% in the earth-moon system:
mu = 0.0122774;
nu = 1 - mu;
syms 'x'
eq = x^5 + (mu - 3)*x^4 + (3 - 2*mu)*x^3 - mu*x^2 + 2 * mu * x - mu == 0;
tmp = double(solve(eq,x));
L = real(tmp(1));

L = 576/1181;


L = [ 0.999 * 881/762
% 5620/6171 %
-1634/1629
% 2819/2640 %
-393/391
0.999 * 1507/1802 ];




ic_L4 = [  1/2 - mu, sqrt(3)/2, 0, 0 ];
ic_L5 = ic_L4;
ic_L5(2) = -ic_L4(2);
ic_L1 = [ L(1), 0, 0, 0 ];

h = 6.336963642732367e-06; % this is the smallest dt from the ode45 results
n_orbit = 2.3;

%% Lagrange points
%{
solver_param = [ "rk4", 1e-6, 1e-5, h, 1 ]; % it is not rk45, just rk4
three_body_problem( ic_L4, t_range_L, n_orbit, mu, "mu", solver_param, 4, true );
%}

%% Lagrange points
%{
for i=1:1:4
    ic_L1 = [ L(i), 0, 0, 0 ];
    solver_param = [ "rk4", 1e-6, 1e-5, h, 1 ];
    three_body_problem( ic_L1, t_range_L, n_orbit, mu, "mu", solver_param, 4, true );
end
%}

%% Lagrange points

solver_param = [ "fehlberg", 1e-6, 1e-5, h, 1 ];
three_body_problem( ic_L5, t_range_L, n_orbit, mu, "mu", solver_param, 4, true );


%% uncomment to solve with all solvers
%{
solver_param = [ "ode45", 1e-6, 1e-5, h, 0 ];
three_body_problem( ic_a, t_range_a, n_orbit, mu, "mu", solver_param, 1, true);
three_body_problem( ic_b, t_range_b, n_orbit, mu, "mu", solver_param, 2, true );
three_body_problem( ic_c, t_range_c, n_orbit, mu, "mu", solver_param, 3, true );
%}

% solver_param = [ "rk4", 1e-6, 1e-5, h, 0 ];
% three_body_problem( ic_a, t_range_a, n_orbit, mu, "mu", solver_param, 4, true );
% three_body_problem( ic_b, t_range_b, n_orbit, mu, "mu", solver_param, 5, true );
% three_body_problem( ic_c, t_range_c, n_orbit, mu, "mu", solver_param, 6, true );

%{
solver_param = [ "fehlberg", 1e-6, 1e-5, h, 0 ];
three_body_problem( ic_a, t_range_a, n_orbit, mu, "mu", solver_param, 7, true );
three_body_problem( ic_b, t_range_b, n_orbit, mu, "mu", solver_param, 8, true );
three_body_problem( ic_c, t_range_c, n_orbit, mu, "mu", solver_param, 9, true );
%}

%% uncomment for plotting of error for multiple tolerances
%{
for j = 16:2:16
    n_orbit = 2.2; % 2.3: sum( abs( yh - y(:,i+1) ) ) = 0 a lot of times: hardly increases h
    solver_param = [ "fehlberg", 10^(-j), 1e-5, h, 0 ];
    % solver_param = [ "ode45", 10^(-j), 10^(-j+1), h, 0 ];
    three_body_problem( ic_c, t_range_c, n_orbit, mu, "mu", solver_param, j, true );
end
%}

%% uncomment for benchmark
% runtime benchmark
%{
solver_param = [ "ode45", 1e-6, 1e-5, h, 0 ];
tic
for j = 1:1:50
    three_body_problem( ic_a, t_range_a, n_orbit, mu, "mu", solver_param, 1, false);
    three_body_problem( ic_b, t_range_b, n_orbit, mu, "mu", solver_param, 2, false );
    three_body_problem( ic_c, t_range_c, n_orbit, mu, "mu", solver_param, 3, false );
end
toc

solver_param = [ "rk4", 1e-6, 1e-5, h, 0 ];
tic
for j = 1:1:50
    three_body_problem( ic_a, t_range_a, n_orbit, mu, "mu", solver_param, 4, false );
    three_body_problem( ic_b, t_range_b, n_orbit, mu, "mu", solver_param, 5, false );
    three_body_problem( ic_c, t_range_c, n_orbit, mu, "mu", solver_param, 6, false );
end
toc

solver_param = [ "fehlberg", 1e-6, 1e-5, h, 0 ];
tic
for j = 1:1:50
    three_body_problem( ic_a, t_range_a, n_orbit, mu, "mu", solver_param, 7, false );
    three_body_problem( ic_b, t_range_b, n_orbit, mu, "mu", solver_param, 8, false );
    three_body_problem( ic_c, t_range_c, n_orbit, mu, "mu", solver_param, 9, false );
end
toc
%}

%%
function output = three_body_problem( ic, t_range, n_orbit, eq_param, mass_or_mu, solver_param, figure_number, plot_fig, h_max )
    arguments
        ic;
        t_range;
        n_orbit;
        eq_param;
        mass_or_mu;
        solver_param;
        figure_number;
        plot_fig;
        h_max  = 0.1; % default value
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ic: array of initial conditions [ u1(0), u2(0), ud1(0), ud2(0), ]
    % t_range: t1: time period of one orbit, [ t0, t1 ]
    % n_orbit: solve for n number of time periods
    % eq_param: mass of bodies: [ m1, m2 ] or
    % eq_param: mu parameter: mu
    % mass_or_mu:   input "mass" if masses are given
    %               input "mu" if the mu parameter is given
    % solver_param: [ solver, abstol, reltol, h ]
    %       solver: choose a solver: "ode45", "rk45", "fehlberg"
    %       abstol: absolut tolerance (or ETOL) ( for ode45 and fehlberg4(5) pair )
    %               fehlberg takes the max of absolute difference between the
    %               elements
    %       reltol: relative tolerance
    %       h: stepsize for rk45
    %       Lagrange: 1 if stationary points are examined
    %                   plot limit set by the trajectory, and not default
    % figure_number: number of figure to plot the orbit on
    % h_max:    maximum of the timestep
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    format long

    % handling input parameters
    solver = solver_param(1);
    abstol = double(solver_param(2));
    reltol = double(solver_param(3));
    h = double( solver_param(4) );
    Lagrange = double( solver_param(5) );
    if mass_or_mu == "mass"
        m1 = eq_param(1);
        m2 = eq_param(2);
        mu = m1 / ( m1 + m2 );
    end
    if mass_or_mu == "mu"
        % in the earth-moon system:
        mu = eq_param;
    end
    nu = 1 - mu;
    
    % solving with built in ode45 solver
    if solver == "ode45"
        opts = odeset( 'AbsTol', abstol, 'RelTol', reltol, 'Stats', 'on' );
        [ t, y ] = ode45( @(t,y) f( y, mu, nu ), n_orbit * t_range, ic, opts );
        tmp = size(t);
        % tmp(1)
        for i=1:1:tmp(1)
            E(i) = e( y(i,:), mu, nu ); 
        end
        if plot_fig
            figure(figure_number)
            hold on
            linewidth = 2;
            plot(y(:,1), y(:,2), 'LineWidth', linewidth)

            if ~Lagrange
                plot(-mu, 0, 'ro', 'MarkerSize', 8, 'LineWidth', linewidth)
                plot(nu, 0, 'ro', 'MarkerSize', 8, 'LineWidth', linewidth)
                axis([-1.5 1.5 -1.5 1.5]);
            end

            title('Numerical approximation', 'FontSize', 16)
            xlabel('u_1', 'FontSize', 14)
            ylabel('u_2', 'FontSize', 14)
            set(gca, 'FontSize', 14)
            
            figure(figure_number + 100)
            hold on
            linewidth = 2;
            plot(t, E/E(1), 'LineWidth', linewidth)
            title('Total Energy', 'FontSize', 16)
            xlabel('t', 'FontSize', 14)
            ylabel('E/E0', 'FontSize', 14)
            set(gca, 'FontSize', 14)

        end % of plotting
    end % of ode45 solver

    % solving with runge-kutta4
    if solver == "rk4"
        disp('rk4');
        N = int64( ( t_range(2) * n_orbit - t_range(1) ) / h ); % stepnumber
        t = linspace( t_range(1), t_range(2) * n_orbit, N+1 ); % grid
        y = zeros( 4, N+1 ); % vector of the numerical approximation
        % computing the numerical approximation using the RK4 method:
        y( :, 1 ) = ic;
        E = zeros( 1, N+1 ); 
        E(1) = e( ic, mu, nu );
        for j = 1:1:N
            y1 = y(:,j);
            k1 = f( y1, mu, nu );
            y2 = y(:,j) + h/2 * k1;
            k2 = f( y2, mu, nu );
            y3 = y(:,j) + h/2*k2;
            k3 = f( y3, mu, nu );
            y4 = y(:,j) + h*k3;
            k4 = f( y4, mu, nu );
            y(:,j+1) = y(:,j) + h/6 *( k1 + 2*k2 + 2*k3 + k4 );
            E(j+1) = e( y(:,j+1), mu, nu );

        end % of rk4
        
        if plot_fig
            % plotting
            figure(figure_number)
            hold on
            %{
            plot( y(1,:), y(2,:) )
            plot(mu, 0, 'ro') % massive bodies
            plot(nu, 0, 'ro') % massive bodies
            axis([-1.5 1.5 -1.5 1.5]);
            title('Numerical approximation')
            %}
            linewidth = 2;
            plot(y(1,:), y(2,:), 'LineWidth', linewidth)
            if ~Lagrange
                plot(-mu, 0, 'ro', 'MarkerSize', 8, 'LineWidth', linewidth)
                plot(nu, 0, 'ro', 'MarkerSize', 8, 'LineWidth', linewidth)
                axis([-1.5 1.5 -1.5 1.5]);
            end
            title('Numerical approximation', 'FontSize', 16)
            xlabel('u_1', 'FontSize', 14)
            ylabel('u_2', 'FontSize', 14)
            set(gca, 'FontSize', 14)

            figure(figure_number + 100)
            hold on
            linewidth = 2;
            plot(t, E/E(1), 'LineWidth', linewidth)
            % axis([-1.5 1.5 -1.5 1.5]);
            title('Total Energy', 'FontSize', 16)
            xlabel('t', 'FontSize', 14)
            ylabel('E/E0', 'FontSize', 14)
            set(gca, 'FontSize', 14)
        end % of plotting
    end % of rk4 solver

    if solver == "fehlberg"
        % data from the Butcher tableau for Fehlberg4(5) pair
        c = [ 0, 1/4, 3/8, 12/13, 1, 1/2 ]; 
        b = [ 25/216, 0, 1408/2565,  2197/4104, -1/5, 0 ];
        bh = [ 16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55 ];
        a = zeros(6); % ( stage index, "coefficient of nth k" index )
        a(2,1) = 1/4;
        a(3,1:2) = [ 3/32, 9/32 ];
        a(4,1:3) = [ 1932/2197, -7200/2197, 7296/2197 ];
        a(5,1:4) = [ 439/216, -8, 3680/513, -845/4104 ];
        a(6,1:5) = [ -8/27, 2, -3544/2565, 1859/4104, -11/40 ];
        
        k = zeros(4,6); % ( index for y components, index for stages )
        y = zeros(4,1); % numerical approximation
        y(:,1) = ic;    % initial conditions
        t = zeros(1,1); % time grid, to be expanded
        t(1) = t_range(1); % set t0
        i = 1;          % initialize index counter
        E = zeros(1,1); 
        E(1) = e( ic, mu, nu );
        H = zeros(1,1); 
        % H(1) = h; 

        % set parameters for error control
        error_not_in_range = true;
        abstolmin = abstol * 0.1;
        % abstolmin = abstol * 0.000001;
        abstolmax = abstol * 10;
        frac = 0.9;
        p = 4;
        error = zeros(1,1);
        
        while t(i) < t_range(2) * n_orbit
            while error_not_in_range
                for j = 1:1:6 % "walk" through the stages
                    Y = y(:,i); % calculating the y value where f will be evaluated, to get the next k value
                    for n = 1:1:j-1
                        Y = Y + h * a(j,n) * k(:,n);
                    end
                    k(:,j) = f( Y, mu, nu );
                end
                y(:,i+1) = y(:,i);

                % 4th order
                for j = 1:1:5
                    y(:,i+1) = y(:,i+1) + h * b(j) * k(:,j);
                end

                % 5th order
                yh = y(:,i);
                for j = 1:1:6
                    yh = yh + h * bh(j) * k(:,j);
                end
                
                % examine error, and manipulate step size accordingly
                % E_change_rate = abs( e( y(:,i), mu, nu ) - e( y(:,i+1), mu, nu ) ) / h;
                % sum( abs( yh - y(:,i+1) ) ) ~= 0
                if max( abs( yh - y(:,i+1) ) ) > abstolmax
                    error_not_in_range = true; % step solution rejected
                    h = ( frac * abstol / sum( abs( yh - y(:,i+1) ) ) ) ^( 1/(p+1) ) * h;
                elseif max( abs( yh - y(:,i+1) ) ) < abstolmin & ( sum( abs( yh - y(:,i+1) ) ) ~= 0 ) % result: no crash, but slow computation
                    error_not_in_range = false; % step solution accepted
                    H(i) = h;
                    h_tmp = ( frac * abstol / sum( abs( yh - y(:,i+1) ) ) ) ^( 1/(p+1) ) * h;
                    if h_tmp <= h_max
                        h = h_tmp;
                    else
                        % h = h;
                    end
                else
                    H(i) = h;
                    error_not_in_range = false;
                end % of error examination
            end % of one iteration step (can be accepted or rejected => repeated)

            error(:,i+1) = max( abs( yh - y(:,i+1) ) );
            t(i+1) = t(i) + h;
            E(i+1) = e( y(:,i+1), mu, nu ); % fix at the othere inputs
            error_not_in_range = true;
            i = i + 1; % i++ not working :(
        end % of fehlberg simulation
        
        if plot_fig
            figure(figure_number)
            hold on
            linewidth = 2;
            plot(y(1,:), y(2,:), 'LineWidth', linewidth)
            if ~Lagrange
                plot(-mu, 0, 'ro', 'MarkerSize', 8, 'LineWidth', linewidth)
                plot(nu, 0, 'ro', 'MarkerSize', 8, 'LineWidth', linewidth)
                axis([-1.5 1.5 -1.5 1.5]);
            end
            title('Numerical approximation', 'FontSize', 16)
            xlabel('u_1', 'FontSize', 14)
            ylabel('u_2', 'FontSize', 14)
            set(gca, 'FontSize', 14)
            
            % plotting error
            figure((figure_number + 1)^6)
            hold on
            plot( t(:), error(:,:) , 'LineWidth', linewidth)
            xlabel('t', 'FontSize', 14)
            ylabel('max(error)', 'FontSize', 14)
            set(gca, 'FontSize', 14)
            title('Numerical approximation error approximation', 'FontSize', 16)

            % plotting the energy of the system
            figure(figure_number + 100)
            hold on
            linewidth = 2;
            plot(t, E/E(1), 'LineWidth', linewidth)
            title('Total Energy', 'FontSize', 16)
            xlabel('t', 'FontSize', 14)
            ylabel('E/E0', 'FontSize', 14)
            set(gca, 'FontSize', 14)

            % plotting the energy change of the system
            figure(figure_number + 200)
            hold on
            linewidth = 2;
            plot( t( 2:(end) ), ( E( 2:end ) - E( 1:(end-1) )  ) ./ H(:).' , 'LineWidth', linewidth)
            title('Total Energy change rate', 'FontSize', 16)
            xlabel('t', 'FontSize', 14)
            ylabel('( E(n) - E(n-1) ) / h', 'FontSize', 14)
            set(gca, 'FontSize', 14)
        end % of plotting
    end % of fehlberg solver
end % solver function

% equation of motion RHS, after Cauchy "átírás" (?eng.?)
function output = f( y, mu, nu )
    d1 = ( ( y(1) + mu )^2 + y(2)^2 )^( 3/2 );
    d2 = ( ( y(1) - nu )^2 + y(2)^2 )^( 3/2 );
    yd1 = y(3);
    yd2 = y(4);
    yd3 = y(1) + 2 * y(4) - nu * ( y(1) + mu ) / d1 - mu * ( y(1) - nu ) / d2;
    yd4 = y(2) - 2 * y(3) - nu * y(2) / d1 - mu * y(2) / d2;
    output = [ yd1; yd2; yd3; yd4 ];
end % RHS

% energy of the system for error checking:
function output = e( y, mu, nu )
    m = 1;  % what is the mass of the small, orbiting object?
            % mabey i can return the energy pre unit mass
    kinetic = 1 / 2 * m * ( y(3)^2 + y(4)^2 );
    h1 = ( ( y(1) - nu )^2 + y(2)^2 )^( 1/2 );
    h2 = ( ( y(1) + mu )^2 + y(2)^2 )^( 1/2 );
    g1 = - mu / h1;         % i have jet to unveil how to calculate the potential energy
                            % is one of the large bodies orbiting the other one? 
    g2 = - nu / h2;         % moreover, i am not even sure if the systems energy should be constant 
    r = -1/2 * ( y(1)^2 + y(2)^2 );
    potential = g1 + g2 + r;
    output = kinetic + potential;
end % system's energy

%{

1   2nd body orbit is circle                                    OK
2   E is constant                                               OK
3   when inceasing the step size, the y value can be accepted   OK
4   examine no solution if error tolerance is too small         OK
5   " I might try to incorporate the min(distance between the   idk
    third and first and between the third and first) into a 
    RelTol like number and incorporate it into the error 
    estimation of the algorithm "
+1  L4 L5 points can be examined                                OK
+2  energy change based step size refinement                    NOT OK
+3  L1 L2 L3                                                    NOT OK

And now for something completly different:

1   add if statement, so the y_hat - y_n+1 = 0 can be handeled          somewhat OK
2   implement h_max, h_min                                              OK
3   choose and implement 1D nonstiff problem, and compare the solvers   OK
4   make step size sequence more smooth                                 
5   redo benchraks                                                      
6   do the documentation                                                
%}