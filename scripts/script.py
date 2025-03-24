import numpy as np
import mcubes
import trimesh
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import time
import os
def cartoon_face(x, y, z, detail=1.0):
    """
    Generate a more stylized cartoon face as a signed distance field.
    """

    # Main head (rounder)
    head = ((x / 1.1)**2 + (y / 1.1)**2 + (z / 1.0)**2) - 1.0

    # Eye sockets (deeper, higher up)
    eye_socket1 = ((x + 0.4)**2 + (y + 0.4)**2 + (z + 0.6)**2) - (0.2 * detail)**2
    eye_socket2 = ((x - 0.4)**2 + (y + 0.4)**2 + (z + 0.6)**2) - (0.2 * detail)**2

    # Pupils (closer to front of eye sockets)
    pupil1 = ((x + 0.4)**2 + (y + 0.4)**2 + (z + 0.7)**2) - (0.07 * detail)**2
    pupil2 = ((x - 0.4)**2 + (y + 0.4)**2 + (z + 0.7)**2) - (0.07 * detail)**2

    # Nose (rounded triangular blob)
    nose = (((x / 0.5)**2 + (z + 0.6)**2 + (y / 1.2)**2) - 0.1**2)

    # Mouth (wide cartoon smile)
    smile = ((x / 0.9)**2 + (y + 0.5)**2 + ((z + 0.8) / 2.0)**2) - (0.15 * detail)**2
    smile = np.where(y < -0.5, smile, 1.0)

    # Ears (move them closer to the head)
    ear1 = (((x + 0.85)**2 + y**2 + z**2) - (0.1 * detail)**2)
    ear2 = (((x - 0.85)**2 + y**2 + z**2) - (0.1 * detail)**2)

    # Blend shapes using CSG operations:
    face = head
    face = np.maximum(face, -eye_socket1)
    face = np.maximum(face, -eye_socket2)
    face = np.minimum(face, pupil1)
    face = np.minimum(face, pupil2)
    face = np.maximum(face, -nose)
    face = np.maximum(face, -smile)
    face = np.minimum(face, ear1)
    face = np.minimum(face, ear2)

    return face

# ====== Step 1: Create a more detailed procedural volume function ======
# def cartoon_face(x, y, z, detail=1.0):
#     """
#     Generate a procedural cartoon face with adjustable detail level.
#     Returns a signed distance field where negative values are inside the shape.
#     """
#     # Head shape (slightly elongated ellipsoid)
#     head = ((x / 1.0)**2 + (y / 1.2)**2 + (z / 1.0)**2) - 1.0
    
#     # Eyes (spherical cavities)
#     eye_size = 0.08 * detail
#     eye1 = ((x + 0.4)**2 + (y + 0.3)**2 + (z + 0.7)**2) - eye_size**2
#     eye2 = ((x - 0.4)**2 + (y + 0.3)**2 + (z + 0.7)**2) - eye_size**2
    
#     # Pupils (smaller spheres inside eyes)
#     pupil_size = 0.04 * detail
#     pupil1 = ((x + 0.4)**2 + (y + 0.3)**2 + (z + 0.78)**2) - pupil_size**2
#     pupil2 = ((x - 0.4)**2 + (y + 0.3)**2 + (z + 0.78)**2) - pupil_size**2
    
#     # Nose (slightly elongated sphere)
#     nose_x = x
#     nose_y = y - 0.1
#     nose_z = z + 0.8
#     nose_shape = ((nose_x)**2 + (nose_y / 1.2)**2 + (nose_z / 0.8)**2) - 0.07**2
    
#     # Smile (toroidal shape)
#     r1, r2 = 0.4, 0.1  # Major and minor radii
#     mouth_x, mouth_y, mouth_z = x, y - 0.4, z + 0.6
#     q = np.sqrt(mouth_x**2 + mouth_z**2) - r1
#     mouth = q**2 + mouth_y**2 - r2**2
#     # Clip mouth to only show bottom half
#     mouth_mask = mouth_y < 0
#     mouth = np.where(mouth_mask, mouth, 1.0)
    
#     # Ears (ellipsoids)
#     ear_size = 0.15 * detail
#     ear1 = ((x + 0.9)**2 + (y)**2 + (z)**2) - ear_size**2
#     ear2 = ((x - 0.9)**2 + (y)**2 + (z)**2) - ear_size**2
    
#     # Combine shapes using CSG operations
#     # Start with the head
#     face = head
    
#     # Subtract eyes
#     face = np.maximum(face, -eye1)
#     face = np.maximum(face, -eye2)
    
#     # Add pupils (union)
#     face = np.minimum(face, pupil1)
#     face = np.minimum(face, pupil2)
    
#     # Subtract nose
#     face = np.maximum(face, -nose_shape)
    
#     # Subtract mouth
#     face = np.maximum(face, -mouth)
    
#     # Union with ears
#     face = np.minimum(face, ear1)
#     face = np.minimum(face, ear2)
    
#     return face

def generate_mesh(resolution=128, detail=1.0, smoothing_iterations=0):
    """
    Generate a mesh from the cartoon face volume with the given resolution and detail.
    Returns vertices, triangles, and the mesh timing information.
    """
    print(f"Generating mesh with resolution {resolution}x{resolution}x{resolution}...")
    
    start_time = time.time()
    
    # Create the volume grid
    X, Y, Z = np.mgrid[-1.5:1.5:resolution*1j, -1.5:1.5:resolution*1j, -1.5:1.5:resolution*1j]
    
    # Generate the signed distance field
    volume = cartoon_face(X, Y, Z, detail)
    
    # Extract mesh with marching cubes
    vertices, triangles = mcubes.marching_cubes(volume, 0)
    
    # Normalize vertices to [-1, 1] range
    vertices = (vertices / resolution) * 3.0 - 1.5
    
    # Apply Laplacian smoothing if requested
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    if smoothing_iterations > 0:
        try:
            # Try the modern approach first
            if hasattr(trimesh.graph, 'smooth_shade'):
                for _ in range(smoothing_iterations):
                    trimesh.graph.smooth_shade(mesh)
            # Fall back to the deprecated method
            else:
                mesh = mesh.smoothed(iterations=smoothing_iterations)
        except Exception as e:
            print(f"Warning: Smoothing failed with error: {e}")
            print("Continuing with unsmoothed mesh...")
    
    elapsed_time = time.time() - start_time
    
    stats = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "time": elapsed_time
    }
    
    return mesh.vertices, mesh.faces, stats, mesh

def visualize_mesh_matplotlib(mesh, output_image="mesh_preview.png", show=True):
    """
    Visualize a mesh using matplotlib, which doesn't require pyglet.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Create a new figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract mesh faces and vertices
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create a Poly3DCollection from the mesh faces
        mesh_collection = Poly3DCollection(vertices[faces], alpha=0.7, linewidths=0.1, edgecolors='k')
        
        # Color the faces using normal directions as RGB
        normals = np.array([np.cross(vertices[face[1]] - vertices[face[0]], 
                                     vertices[face[2]] - vertices[face[0]]) 
                            for face in faces])
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        colors = (normals + 1) / 2  # Convert from [-1, 1] to [0, 1]
        mesh_collection.set_facecolor(colors)
        
        # Add the collection to the axes
        ax.add_collection3d(mesh_collection)
        
        # Set axes limits based on mesh bounds
        bounds = mesh.bounds
        ax.set_xlim(bounds[0, 0], bounds[1, 0])
        ax.set_ylim(bounds[0, 1], bounds[1, 1])
        ax.set_zlim(bounds[0, 2], bounds[1, 2])
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set title
        ax.set_title('Cartoon Face 3D Mesh')
        
        # Use equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save the figure
        plt.savefig(output_image, dpi=200, bbox_inches='tight')
        print(f"✅ Saved mesh preview image to {output_image}")
        
        # Show the figure if requested
        if show:
            plt.show()
            
    except Exception as e:
        print(f"Warning: Matplotlib visualization failed with error: {e}")
        print("Try installing the required packages with: pip install matplotlib")

def show_mesh(mesh, output_image="mesh_preview.png"):
    """
    Try different methods to visualize the mesh, falling back as needed.
    """
    # Try using trimesh's viewer first
    try:
        mesh.show()
        return True
    except ImportError as e:
        print(f"Trimesh viewer error: {e}")
        print("Installing pyglet<2 would fix this, but trying alternative visualization...")
    except Exception as e:
        print(f"Trimesh viewer failed with error: {e}")
        print("Trying alternative visualization...")
    
    # Fall back to matplotlib visualization
    return visualize_mesh_matplotlib(mesh, output_image=output_image)

def show_volume_slices(resolution=128, detail=1.0):
    """
    Show three orthogonal slices through the volume for better understanding
    of the signed distance field.
    """
    # Create the volume grid
    X, Y, Z = np.mgrid[-1.5:1.5:resolution*1j, -1.5:1.5:resolution*1j, -1.5:1.5:resolution*1j]
    volume = cartoon_face(X, Y, Z, detail)
    
    # Create a figure with 3 subplots for XY, XZ, and YZ planes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Volume Slices (Signed Distance Field)", fontsize=16)
    
    # Get the middle slices
    mid_z = volume.shape[2] // 2
    mid_y = volume.shape[1] // 2
    mid_x = volume.shape[0] // 2
    
    # XY plane (constant Z)
    xy_slice = volume[:, :, mid_z]
    axes[0].imshow(xy_slice.T, origin='lower', cmap='RdBu', 
                 extent=[-1.5, 1.5, -1.5, 1.5])
    axes[0].contour(xy_slice.T, levels=[0], colors='black', 
                  extent=[-1.5, 1.5, -1.5, 1.5])
    axes[0].set_title(f"XY Plane (Z={0:.2f})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    
    # XZ plane (constant Y)
    xz_slice = volume[:, mid_y, :]
    axes[1].imshow(xz_slice.T, origin='lower', cmap='RdBu', 
                 extent=[-1.5, 1.5, -1.5, 1.5])
    axes[1].contour(xz_slice.T, levels=[0], colors='black', 
                  extent=[-1.5, 1.5, -1.5, 1.5])
    axes[1].set_title(f"XZ Plane (Y={0:.2f})")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    
    # YZ plane (constant X)
    yz_slice = volume[mid_x, :, :]
    axes[2].imshow(yz_slice.T, origin='lower', cmap='RdBu', 
                 extent=[-1.5, 1.5, -1.5, 1.5])
    axes[2].contour(yz_slice.T, levels=[0], colors='black', 
                  extent=[-1.5, 1.5, -1.5, 1.5])
    axes[2].set_title(f"YZ Plane (X={0:.2f})")
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")
    
    # Save the figure
    slice_image = "volume_slices.png"
    plt.tight_layout()
    plt.savefig(slice_image, dpi=200, bbox_inches='tight')
    print(f"✅ Saved volume slices to {slice_image}")
    
    plt.show()

def save_mesh_views(mesh, output_prefix="mesh_view"):
    """
    Save multiple views of the mesh from different angles.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Define viewing angles (azimuth, elevation)
        views = [
            (0, 0),    # Front view
            (90, 0),   # Right side
            (-90, 0),  # Left side
            (180, 0),  # Back view
            (0, 90),   # Top view
            (0, -90),  # Bottom view
        ]
        
        view_names = ["front", "right", "left", "back", "top", "bottom"]
        
        # Extract mesh data
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Calculate face colors based on normals
        normals = np.array([np.cross(vertices[face[1]] - vertices[face[0]], 
                                   vertices[face[2]] - vertices[face[0]]) 
                          for face in faces])
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        colors = (normals + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Create a directory for the views if it doesn't exist
        views_dir = "mesh_views"
        os.makedirs(views_dir, exist_ok=True)
        
        # Generate each view
        for i, ((azim, elev), name) in enumerate(zip(views, view_names)):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create mesh collection
            mesh_collection = Poly3DCollection(vertices[faces], alpha=0.9, linewidths=0.2, edgecolors='k')
            mesh_collection.set_facecolor(colors)
            ax.add_collection3d(mesh_collection)
            
            # Set viewing angle
            ax.view_init(elev=elev, azim=azim)
            
            # Set axes limits based on mesh bounds
            bounds = mesh.bounds
            ax.set_xlim(bounds[0, 0], bounds[1, 0])
            ax.set_ylim(bounds[0, 1], bounds[1, 1])
            ax.set_zlim(bounds[0, 2], bounds[1, 2])
            
            # Remove axis ticks and labels for cleaner rendering
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            
            # Set title
            ax.set_title(f"{name.capitalize()} View")
            
            # Use equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Save the figure
            output_file = os.path.join(views_dir, f"{output_prefix}_{name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        print(f"✅ Saved multiple mesh views to the {views_dir}/ directory")
        
    except Exception as e:
        print(f"Warning: Multi-view rendering failed with error: {e}")

# ====== Step 2: Command-line interface with visualization options ======
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a cartoon face mesh using marching cubes')
    parser.add_argument('--resolution', type=int, default=128, help='Grid resolution (default: 128)')
    parser.add_argument('--detail', type=float, default=1.0, help='Detail level (default: 1.0)')
    parser.add_argument('--output', type=str, default='cartoon_face.obj', help='Output OBJ file (default: cartoon_face.obj)')
    parser.add_argument('--smoothing', type=int, default=0, help='Number of smoothing iterations (default: 0)')
    parser.add_argument('--visualize', action='store_true', help='Show 3D visualization (default: False)')
    parser.add_argument('--slices', action='store_true', help='Show volume slices (default: False)')
    parser.add_argument('--views', action='store_true', help='Save multiple views of the mesh (default: False)')
    args = parser.parse_args()
    
    # Generate the mesh
    vertices, triangles, stats, mesh = generate_mesh(
        resolution=args.resolution, 
        detail=args.detail,
        smoothing_iterations=args.smoothing
    )
    
    # Save to OBJ file
    mcubes.export_obj(vertices, triangles, args.output)
    print(f"✅ Saved mesh to {args.output}")
    print(f"   Vertices: {stats['vertices']}")
    print(f"   Triangles: {stats['triangles']}")
    print(f"   Generation time: {stats['time']:.2f} seconds")
    
    # Save multiple views of the mesh if requested
    if args.views:
        save_mesh_views(mesh)
    
    # Show volume slices if requested
    if args.slices:
        show_volume_slices(args.resolution, args.detail)
    
    # Show 3D visualization if requested
    if args.visualize:
        # Calculate vertex colors based on normals
        colors = np.zeros((len(mesh.vertices), 4))
        normals = mesh.vertex_normals
        # Map normals to RGB colors (normalized from -1,1 to 0,1)
        colors[:, 0:3] = (normals + 1.0) / 2.0
        colors[:, 3] = 1.0  # Full alpha
        mesh.visual.vertex_colors = colors
        
        # Try to show the mesh
        show_mesh(mesh)

# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()