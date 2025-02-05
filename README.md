# fsi
Fluid-Structure-Interaction in deal.II
-

This github version is based on 

https://media.archnumsoft.org/10305/

https://doi.org/10.11588/ans.2013.1.10305 

and solves the fluid-structure interaction benchmark problems FSI 1,2,3.

The underlying open-source software library is 

https://www.dealii.org

The latest tested version of this code is based on deal.II version 9.6.0 (Feb 2025)

Credit and citation
-

If you use this code, older versions, or parts, it would be nice to give credit. A bibtex entry is

@article{Wi13_fsi_with_deal,<br/>
   author = {T. Wick},<br/>
   title = {Solving Monolithic Fluid-Structure Interaction Problems in Arbitrary 
            {L}agrangian {E}ulerian Coordinates with the deal.{II} Library},
   journal = {Archive of Numerical Software},<br/>
   year = {2013},<br/>
   volume = {1},<br/>
   pages = {1-19},<br/>
   url = "https://media.archnumsoft.org/10305/",<br/>
   doi = "https://doi.org/10.11588/ans.2013.1.10305" <br/>
}



General description
-
We describe a setting of a nonlinear fluid-structure interaction problem and the corresponding solution process in the finite element software package deal.II. The fluid equations are transformed via the ALE mapping (Arbitrary Lagrangian Eulerian framework) to a reference configuration and these are coupled with the structure equations by a monolithic solution algorithm. To construct the ALE mapping, we use a biharmonic equation. Finite differences are used for temporal discretization. The derivation is realized in a general manner that serves for different time stepping schemes. Spatial discretization is based on a Galerkin finite element scheme. The nonlinear system is solved by a Newton method. Using this approach, the Jacobian matrix is constructed by exact computation of the directional derivatives. The implementation using the software library package deal.II serves for the computation of different fluid-structure configurations. Specifically, our geometry data are taken from the fluid-structure benchmark configuration that was proposed in 2006 in the DFG project Fluid-Structure Interaction I: Modelling, Simulation, Optimisation. Our results show that this implementation using deal.II is able to produce comparable findings. 
