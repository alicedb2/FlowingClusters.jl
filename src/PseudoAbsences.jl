module PseudoAbsences
    using NearestNeighbors: BallTree, inrangecount
    using Distances: Minkowski
    using Distributions: MvNormal
    using StatsBase: median
    using LinearAlgebra: norm, diagm

    function env_pseudoabsences(samples, exclusion_radius, nb_pseudoabsences; region_factor=1.5, shape=:ball, verbose=false)

        d = size(samples, 1)

        nb_data_points = size(samples, 2)
            
        if shape == :ball
            # For the "ball" method find the region
            # around the median of presences which is
            # region_factor times larger than the smallest
            # hyperball centered at the medians along each dimensions.
            # We will then do uniform rejection sampling of
            # pseudoabsences within that hyperball.
            center_x = median(samples, dims=2)[:]
            max_radius_around_center = maximum(norm.(eachcol(samples .- center_x)))
            region_radius = region_factor * max_radius_around_center
        elseif shape == :box
            # For the "box" method we find the region
            # that is region_factor times larger than the smallest
            # hyperbox centered at the the middles points between
            # the furthest presences along each dimensions.
            # We will then do uniform rejection sampling of
            # pseudoabsences within that hyperbox.
            mins = [minimum(samples[i, :]) for i in 1:d]
            maxs = [maximum(samples[i, :]) for i in 1:d]
            centers = [(ma + mi)/2 for (ma, mi) in zip(maxs, mins)]
            widths = region_factor * [(ma - mi) for (ma, mi) in zip(maxs, mins)]
        end
        # Finding the bounding hyperellipse would be really nice
        # but finding such an hyperellipse is very much non-trivial

        
        balltree = BallTree(samples, Minkowski(2))

        pseudoabsences = zeros(d, nb_pseudoabsences)
        
        hits = 0
        misses = 0
        i = 0
        
        while true
            if shape == :ball
                dist = MvNormal(zeros(d), diagm(fill(1, d)))
                # Uniform sampling within an hyperball
                u = rand(dist)
                nor = norm(u)
                r = region_radius * sqrt(rand())
                x = r .* u ./ nor + center_x
            elseif shape == :box
                # Uniform sampling within an hyperbox
                u = widths .* (rand(d) .- 1/2)
                x = centers .+ u
            end
            
            # This is the rejection sampling step. We just sample
            # uniformly within the hyperball/box. The point must be
            # at least exclusion_radius further away than the closest
            # presence otherwise it is rejected and we try again
            if inrangecount(balltree, x, exclusion_radius) == 0
                hits += 1
                pseudoabsences[:, hits] = x
            else
                misses += 1
            end

            i += 1
            
            a = (mod(i, 1000) == 0)
            b = (mod(i, 1000) == 0)
            c = (i == 0)
            e = (hits == nb_pseudoabsences)
                
            if verbose && (a || b || c || e)
                print("\r$hits hits ($misses misses)")
                flush(stdout)
            end
            
            if hits >= nb_pseudoabsences
                break
            end
            
            if misses/(hits + 1) >= 5000.0
                error("Generating 10,000x more misses than hits. Try increasing region_factor or decreasing exclusion_radius.")
                flush(stderr)
            end
            
        end    
            
        println()
        
        return pseudoabsences
        
    end
end