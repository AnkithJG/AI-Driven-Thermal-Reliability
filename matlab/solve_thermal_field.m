function T_grid = solve_thermal_field(k_s, k_c, ref)

%=========================================================================
% FEA/FEM Driver code for the heat conduction problem
%=========================================================================

DomainSize = 60e-3;   % half of the rectangle width, one fourth of the rectangle length.
q=6/(0.03*0.03*0.000625);  %power is 6W, volume of cpu is 30mm by 30mm by 0.625mm, 

[ Nodes,TRI, Dir_L, Dir_R, NeumNodes] = rec(DomainSize,ref);

%=== Determine the number of nodes and elements in the domain ===
NNodes = length(Nodes);
NElem = length(TRI);

%=== Initialize the A-matrix and RHS ===
A = spalloc(NNodes, NNodes, 12*NNodes);
F = zeros(NNodes, 1);

%=== cycle through the elements to build the A-matrix and f-vector ===
for(ie=1:NElem)
    element_number = ie;
    N1 = TRI(ie,1); N2 = TRI(ie,2); N3 = TRI(ie,3);
    X1 = Nodes(N1,1); X2 = Nodes(N2,1); X3 = Nodes(N3,1);
    Y1 = Nodes(N1,2); Y2 = Nodes(N2,2); Y3 = Nodes(N3,2);
    
    [TriArea] = calculateArea(X1, X2, X3, Y1, Y2, Y3);

    Plane_Mat = [X1 Y1 1; X2 Y2 1; X3 Y3 1];
    RHS1 = [1 0 0]'; RHS2 = [0 1 0]'; RHS3 = [0 0 1]';
    abc1 = Plane_Mat\RHS1; abc2 = Plane_Mat\RHS2; abc3 = Plane_Mat\RHS3;
    a1 = abc1(1); b1 = abc1(2); c1 = abc1(3);
    a2 = abc2(1); b2 = abc2(2); c2 = abc2(3);
    a3 = abc3(1); b3 = abc3(2); c3 = abc3(3);                    

    if X1<DomainSize/4&&X1>-DomainSize/4&&X2<DomainSize/4&&X2>-DomainSize/4&&Y1<DomainSize/4&&Y1>-DomainSize/4&&Y2<DomainSize/4&&Y2>-DomainSize/4;
        A_elemental = k_s*TriArea*[ a1*a1 + b1*b1  a2*a1 + b2*b1  a3*a1 + b3*b1;    
                                a1*a2 + b1*b2  a2*a2 + b2*b2  a3*a2 + b3*b2; 
                                a1*a3 + b1*b3  a2*a3 + b2*b3  a3*a3 + b3*b3]; 
        F_elemental = TriArea/3*q*[1 1 1];
    else
        A_elemental = k_c*TriArea*[ a1*a1 + b1*b1  a2*a1 + b2*b1  a3*a1 + b3*b1;    
                                a1*a2 + b1*b2  a2*a2 + b2*b2  a3*a2 + b3*b2; 
                                a1*a3 + b1*b3  a2*a3 + b2*b3  a3*a3 + b3*b3];  
        F_elemental = 0*[1 1 1];
    end
    
    A(N1, N1) = A(N1, N1) +  A_elemental(1, 1); A(N2, N1) = A(N2, N1) + A_elemental(2, 1); A(N3, N1) = A(N3, N1) + A_elemental(3, 1); 
    A(N1, N2) = A(N1, N2) + A_elemental(1, 2); A(N2, N2) = A(N2, N2) + A_elemental(2, 2); A(N3, N2) = A(N3, N2) + A_elemental(3, 2); 
    A(N1, N3) = A(N1, N3) + A_elemental(1, 3); A(N2, N3) = A(N2, N3) + A_elemental(2, 3); A(N3, N3) = A(N3, N3) + A_elemental(3, 3); 
    F(N1) = F(N1) + F_elemental(1); F(N2) = F(N2) + F_elemental(2); F(N3) = F(N3) + F_elemental(3);
end

%=== Boundary conditions ===
for(i=1:length(Dir_L))
    A(Dir_L(i),:) = 0; A(Dir_L(i),Dir_L(i)) = 1; F(Dir_L(i)) = 20;
end
for(i=1:length(Dir_R))
    A(Dir_R(i),:) = 0; A(Dir_R(i),Dir_R(i)) = 1; F(Dir_R(i)) = 20; 
end

%=== Solve the problem ===
Sol = A\F;

%=== Interpolation onto grid ===
x_query = linspace(-DomainSize/2, DomainSize/2, 64);
y_query = linspace(-DomainSize/2, DomainSize/2, 64);
[X_grid, Y_grid] = meshgrid(x_query, y_query);
interpolator = scatteredInterpolant(Nodes(:,1), Nodes(:,2), Sol, 'linear', 'nearest');
T_grid = interpolator(X_grid, Y_grid);

end % End of solve_thermal_field

%================ Helper Functions ================
function [TriArea] = calculateArea(X1, X2, X3, Y1, Y2, Y3)
    S1 = ([(X2-X1), (Y2-Y1), 0]);
    S2 = ([(X3-X2), (Y3-Y2), 0]);
    TriArea = norm(cross(S1, S2))/2;
end

function [tnodes, telements, DirE_L, DirE_R, NeumE]=rec(DomainSize,ref)
    model = createpde;
    width = DomainSize/2;
    totalLength = width;
    f=width;

    R1 = [3 4 -totalLength  totalLength totalLength -totalLength -width -width width width]';
    C1 = [3 4 -totalLength/2  totalLength/2 totalLength/2 -totalLength/2 -width/2 -width/2 width/2 width/2]';
         
    for i=1:10
        m(i)=0.2/i*f;
    end
    hmax=m(ref);
     
    gdm = [R1 C1];
    ns = char('R1','C1');
    g = decsg(gdm,'R1 + C1',ns');
    geometryFromEdges(model,g);
     
    % COMMENTED OUT PLOTTING
    % pdegplot(model,'EdgeLabels','on')
    
    MF=generateMesh(model,'Hmax',hmax,'Hmin',0.00000000001,'GeometricOrder','linear','Hgrad',2);
    
    % COMMENTED OUT PLOTTING
    % figure
    % pdemesh(model);

    E1 = findNodes(MF,'region','Edge',1);
    E2 = findNodes(MF,'region','Edge',2);
    E3 = findNodes(MF,'region','Edge',3);
    E4 = findNodes(MF,'region','Edge',4);
    E5 = findNodes(MF,'region','Edge',5);
    E6 = findNodes(MF,'region','Edge',6);
    E7 = findNodes(MF,'region','Edge',7);
    E8 = findNodes(MF,'region','Edge',8);

    DirE_L=[E7]';
    DirE_R=[E2]';
    NeumE=[E1,E3,E4,E5,E6,E8]';
    tnodes=MF.Nodes';
    telements=MF.Elements';
end
