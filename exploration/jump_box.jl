function rot_SO2(theta)
   rot = [cos(theta) sin(theta); -sin(theta) cos(theta)]
   return rot
end

using JuMP

import JSON

"""given symbols, produce a dictionary of symbol=>value"""
macro symbol_dict(symbols...)
	quote
		Dict(
			$(
				[:($(QuoteNode(sym)) => $(esc(sym))) for sym in symbols]...
			)
		)
	end
end


"""given JuMP decision variables, produce a dictionary of symbols=>optimal value"""
macro value_dict(vars...)
	rebindings = [:($(esc(var)) = getValue($(esc(var)))) for var in vars]
	quote
		let $(:($rebindings)...)
			@symbol_dict($([esc(var) for var in vars]...))
		end
	end
end


vertices_zero = zeros(4,2)
half_height = 2
vertices_zero[1,:] = [1,half_height]
vertices_zero[2,:] = [1,-half_height]
vertices_zero[3,:] = [-1,-half_height]
vertices_zero[4,:] = [-1,half_height]

(m1, m2) = (-1.0, -1.0) #velocity in displacement / time
(b1, b2) = (5.5, 5.0)	#starting position

using PyPlot

ax = PyPlot.gca()
ax[:set_aspect]("equal")

function sim(t_max::Float64, ax=Void)
	# simulate physics from time 0 to t_max
	m = Model()

	t0 = 0
	angle_0 = deg2rad(-45.0)
	starting_position = [m1*t0 + b1 m2*t0 + b2]

	vertices_posed = vertices_zero * rot_SO2(angle_0) .+ starting_position #best-effort broadcast semantics?

	c0 = cos(angle_0)
	s0 = sin(angle_0)
	x0 = starting_position[1]
	y0 = starting_position[2]

	vx0 = Float64[vertices_posed[i, 1] for i=1:4]
	vy0 = Float64[vertices_posed[i, 2] for i=1:4]

	@defVar(m, t, start=t0)

	@setObjective(m, Max, t)	

	@defVar(m, x, start=x0)
	@defVar(m, y, start=y0)
	@defVar(m, c, start=c0)
	@defVar(m, s, start=s0)
	@defVar(m, vx[1:4])
	@defVar(m, vy[1:4])

	setValue(vx, vx0)
	setValue(vy, vy0)


	@addConstraint(m, x-m1*t-b1==0)
	@addConstraint(m, y-m2*t-b2==0)

	@addNLConstraint(m, s^2 + c^2 == 1)

	# first row of rotation matrix multiplication
	for i in 1:4 @addConstraint(m, c*vertices_zero[i,1] + -s*vertices_zero[i,2] + x == vx[i] ) end

	# second row of rotation matrix multiplication
	for i in 1:4 @addConstraint(m, s*vertices_zero[i,1] +  c*vertices_zero[i,2] + y == vy[i] ) end

	#y coordinates don't penetrate bottom
	for i in 1:4 @addConstraint(m, vy[i]>=0) end

	#x coordinates don't penetrate left 
	for i in 1:4 @addConstraint(m, vx[i]>=0) end

	#simulate until
	@addConstraint(m, t<=t_max)
	solve(m)

	#lexicographic method
	@addConstraint(m, t==getValue(t))
	@setObjective(m, Max, s*s0 + c*c0) # after all other objectives are met, also rotate the least amount possible.
	solve(m)

	if ax != Void
		#plot starting position of box
		ax[:plot]( vcat(vx0, vx0[1]),  vcat(vy0, vy0[1]), "--")

		#center of starting position, and constraint axis/direction
		ax[:plot](x0,  y0, "o")
		tf = 10
		ax[:plot]([x0, x0+m1*tf], [y0, y0+m2*tf], "--")

		#plot resting position of box
		let vx=getValue(vx), vy=getValue(vy)
			ax[:plot]( vcat(vx, vx[1]),  vcat(vy, vy[1]), "-")
		end

		#center of resting position
		let x=getValue(x), y=getValue(y)
			ax[:plot](x,  y, "o")
		end

		ax[:plot]([0.0, 10.0], [0.0, 0.0], lw=2, color="k")
		ax[:plot]([0.0, 0.0], [0.0, 10.0], lw=2, color="k")
	end


	return @value_dict(t, vx, vy, x, y, c, s)

end


final_hit_time = sim(Inf)[:t]
dicts = [sim(t) for t in linspace(0.0, final_hit_time, 10)]

open("geometric_box_corner.json", "w") do file
	write(file, JSON.json(dicts, 2))
end
