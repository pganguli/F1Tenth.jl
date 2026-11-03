# Compile the common path (load a map, reset, step) ahead of time so the first
# env.step! in a session is fast.
@setup_workload begin
    poses = [0.0 0.0 0.0; 2.0 0.0 0.0]
    @compile_workload Logging.with_logger(Logging.NullLogger()) do
        env = F110Env(map="berlin", num_agents=2)
        reset!(env, poses)
        for _ in 1:3
            step!(env, [0.1 2.0; -0.1 1.0])
        end
        euler = F110Env(map="berlin", num_agents=1, integrator=Integrator.Euler)
        reset!(euler, poses[1:1, :])
        step!(euler, [0.0 1.0])
    end
end
