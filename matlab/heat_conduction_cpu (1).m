function heat_conduction_cpu

close all;
clear all;

%=========================================================================
% FEA/FEM Driver code for the heat conduction problem
%=========================================================================

% Shape =2;    %1 for circle; 2 for ecllipse.
DomainSize = 60e-3;   % half of the rectangle width, one fourth of the rectangle length.
ref = 1;        % <------ Change refinement, 1 to 10
q=6/(0.03*0.03*0.000625);  %power is 6W, volume of cpu is 30mm by 30mm by 0.625mm, 
k_s=140;  % silicon thermal conductivity is 140W/m/K
k_c=387.6;  % copper thermal conductivity is 387.6W/m/K

[ Nodes,TRI, Dir_L, Dir_R, NeumNodes] = rec(DomainSize,ref);

% plot the triangulation
%===
%TRI 384*3 matrix, each row represent the node number of each element
%Nodes 221*2 matrix, each row represent the coordinate of each node, in
%total, there are 221 nodes.
trimesh(TRI, Nodes(:,1),Nodes(:,2))
axis equal
pause(0.1)

%===
% Determine the number of nodes and elements in the domain
%===
NNodes = length(Nodes);
NElem = length(TRI);

%===
% Initialize the A-matrix and RHS to zero valued containers (could use
% spalloc)
%===
A = spalloc(NNodes, NNodes, 12*NNodes);
F = zeros(NNodes, 1);

%===
% cycle through the elements to build the A-matrix and f-vector
%===
for(ie=1:NElem)
    element_number = ie;

    N1 = TRI(ie,1);
    N2 = TRI(ie,2);
    N3 = TRI(ie,3);
    
    X1 = Nodes(N1,1);
    X2 = Nodes(N2,1);
    X3 = Nodes(N3,1);
    
    Y1 = Nodes(N1,2);
    Y2 = Nodes(N2,2);
    Y3 = Nodes(N3,2);
    
    [TriArea] = calculateArea(X1, X2, X3, Y1, Y2, Y3);      % Determine how to calculate and use area

    % Determine how to compute the planar coefficients
    % Solve the matrix of basis function nodes
       Plane_Mat = [X1 Y1 1; X2 Y2 1; X3 Y3 1];
        RHS1 = [1 0 0]';
        RHS2 = [0 1 0]';
        RHS3 = [0 0 1]';

        abc1 = Plane_Mat\RHS1;
        abc2 = Plane_Mat\RHS2;
        abc3 = Plane_Mat\RHS3;

        a1 = abc1(1);
        b1 = abc1(2);
        c1 = abc1(3);

        a2 = abc2(1);
        b2 = abc2(2);
        c2 = abc2(3);

        a3 = abc3(1);
        b3 = abc3(2);
        c3 = abc3(3);                    
%     gN1gN1 = 2;  % Fill the right values
%     gN2gN2 = 2;  % Fill the right values
%     gN3gN3 = 2;  % Fill the right values
% 
%     gN1gN2 = 1;  % Fill the right values
%     gN1gN3 = 1;  % Fill the right values             
%     gN2gN3 = 1;  % Fill the right values
    %[2 1 1;1 2 1;1 1 2]

               
    if X1<DomainSize/4&&X1>-DomainSize/4&&X2<DomainSize/4&&X2>-DomainSize/4&&Y1<DomainSize/4&&Y1>-DomainSize/4&&Y2<DomainSize/4&&Y2>-DomainSize/4;
        A_elemental = k_s*TriArea*[ a1*a1 + b1*b1  a2*a1 + b2*b1  a3*a1 + b3*b1;    
                                a1*a2 + b1*b2  a2*a2 + b2*b2  a3*a2 + b3*b2; 
                                a1*a3 + b1*b3  a2*a3 + b2*b3  a3*a3 + b3*b3]; 
        
        
        F_elemental = TriArea/3*q*[1 1 1];    % Fill the RHS
    else
        
        A_elemental = k_c*TriArea*[ a1*a1 + b1*b1  a2*a1 + b2*b1  a3*a1 + b3*b1;    
                                a1*a2 + b1*b2  a2*a2 + b2*b2  a3*a2 + b3*b2; 
                                a1*a3 + b1*b3  a2*a3 + b2*b3  a3*a3 + b3*b3];  
        
        
        F_elemental = 0*[1 1 1];    % Fill the RHS
    end
   
    
    A(N1, N1) = A(N1, N1) +  A_elemental(1, 1); % Fill the right values
    A(N2, N1) = A(N2, N1) + A_elemental(2, 1); % Fill the right values
    A(N3, N1) = A(N3, N1) + A_elemental(3, 1); % Fill the right values

    A(N1, N2) = A(N1, N2) + A_elemental(1, 2); % Fill the right values
    A(N2, N2) = A(N2, N2) + A_elemental(2, 2); % Fill the right values
    A(N3, N2) = A(N3, N2) + A_elemental(3, 2); % Fill the right values
    
    A(N1, N3) = A(N1, N3) + A_elemental(1, 3); % Fill the right values
    A(N2, N3) = A(N2, N3) + A_elemental(2, 3); % Fill the right values
    A(N3, N3) = A(N3, N3) + A_elemental(3, 3); % Fill the right values

    F(N1) = F(N1) + F_elemental(1); % Fill the right values
    F(N2) = F(N2) + F_elemental(2); % Fill the right values
    F(N3) = F(N3) + F_elemental(3); % Fill the right values
end

%===
% Boundary conditions
%===
for(i=1:length(Dir_L))
    A(Dir_L(i),:) = 0;% Fill the right values
    A(Dir_L(i),Dir_L(i)) = 1;% Fill the right values
    
    F(Dir_L(i)) = 20; % Fill the right values
end

for(i=1:length(Dir_R))
    A(Dir_R(i),:) = 0;% Fill the right values
    A(Dir_R(i),Dir_R(i)) = 1;% Fill the right values
    
    F(Dir_R(i)) = 20; % Fill the right values
end


%===
% Solve the problem
%===
Solution = A\F;
Sol = Solution;


%===
% Post Processing calculations: Don't Mess with what works :-)
%===


%===
% Gradient calculation
%===
for(ie=1:NElem)
    element_number = ie;

    N1 = TRI(ie,1);
    N2 = TRI(ie,2);
    N3 = TRI(ie,3);
    
    X1 = Nodes(N1,1);
    X2 = Nodes(N2,1);
    X3 = Nodes(N3,1);
    
    Y1 = Nodes(N1,2);
    Y2 = Nodes(N2,2);
    Y3 = Nodes(N3,2);

    S1 = Sol(N1);
    S2 = Sol(N2);
    S3 = Sol(N3);

    C = [1 X1 Y1; 1 X2 Y2; 1 X3 Y3]\[eye(3)];
    %uj*delta(phij)
    Gradient_IE(ie,:) = [(S1*C(2,1)+S2*C(2,2)+S3*C(2,3)), ...
        (S1*C(3,1)+S2*C(3,2)+S3*C(3,3))];
    
    Centroid(ie,:) = [(X1+X2+X3)/3, (Y1+Y2+Y3)/3];
end

%===
% Figure 3
%===
figure
trisurf(TRI, Nodes(:,1), Nodes(:,2), Sol) 
hold on
colormap(jet)
shading interp
title('Temperature Profiles')
view([0 0 1])
axis equal
cb = colorbar;                                     % create and label the colorbar
cb.Label.String = 'Temp';

%===
% Figure 4
%===
figure
Hflux = ((Gradient_IE(:,1).^2 + Gradient_IE(:,2).^2).^(.5))';
a = trisurf(TRI, Nodes(:,1), Nodes(:,2), Nodes(:,2)*0, Hflux)
hold on


colormap(jet)
 set(a,'edgealpha',0)
title('Heat Flux Distribution (Centroidal)')
view([0 0 1])
axis equal
cb = colorbar;                                     % create and label the colorbar
cb.Label.String = 'Heat flux';

end





function [TriArea] = calculateArea(X1, X2, X3, Y1, Y2, Y3);

S1 = ([(X2-X1), (Y2-Y1), 0]);
S2 = ([(X3-X2), (Y3-Y2), 0]);

TriArea = norm(cross(S1, S2))/2;

end



function [tnodes telements DirE_L DirE_R NeumE]=rec(DomainSize,ref)



% DomainSize = 60e-3;   % half of the rectangle width, one fourth of the rectangle length.
% ref = 1;        % <------ Change refinement, 1 to 4

model = createpde;




width = DomainSize/2;
totalLength = width;

f=width;

R1 = [3 4 -totalLength  totalLength ...
           totalLength -totalLength ...
          -width -width width width]';

C1 = [3 4 -totalLength/2  totalLength/2 ...
           totalLength/2 -totalLength/2 ...
          -width/2 -width/2 width/2 width/2]';

 %ref is the refinement factor, the higher ref is, the finer the mesh will be.     
     
 for i=1:10
     m(i)=0.2/i*f;
 end
 
 hmax=m(ref);
 
 
gdm = [R1 C1];
ns = char('R1','C1');
g = decsg(gdm,'R1 + C1',ns');



 %geometryFromMesh(model,tnodes,telements);
 geometryFromEdges(model,g);
 
 pdegplot(model,'EdgeLabels','on')
%  figure
%  pdemesh(model)
% 
% figure
% pdemesh(model)

MF=generateMesh(model,'Hmax',hmax,'Hmin',0.00000000001,'GeometricOrder','linear','Hgrad',2);

figure
pdemesh(model);

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
