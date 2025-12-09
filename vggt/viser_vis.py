import time
import argparse
import trimesh
import viser
import numpy as np
from pathlib import Path

def main():
    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------
    # I have hardcoded your path here so you don't need to type it in the terminal.
    import argparse
    args = argparse.Namespace()
    
    # Your specific data path
    args.input = "/projects/standard/csci5561/shared/G11/my_datasets" 
    
    args.port = 8080
    args.no_share = False # Set to True if you want to block the public link

    # -------------------------------------------------------------------------
    # LOGIC
    # -------------------------------------------------------------------------
    dataset_path = Path(args.input)
    
    # Try to find point clouds. 
    # Since 'my_datasets' has subfolders (dishes, etc), we need to look deeper.
    # This logic aggregates ALL ply files from ALL sub-datasets into one list.
    ply_files = []
    if dataset_path.exists():
        # Look in direct outputs
        ply_files.extend(list((dataset_path / "outputs" / "pointclouds").glob("*.ply")))
        
        # Look in subfolders (e.g. dishes/outputs/pointclouds)
        for sub in dataset_path.iterdir():
            if sub.is_dir():
                sub_ply = sub / "outputs" / "pointclouds"
                if sub_ply.exists():
                    ply_files.extend(list(sub_ply.glob("*.ply")))
    
    if not ply_files:
        print(f"Error: No .ply files found in {dataset_path} or its subfolders.")
        print("Did you run the processing script (vggt_robust.py) first?")
        return

    print(f"---------------------------------------------------------")
    print(f"Found {len(ply_files)} point clouds.")
    print(f"Starting Viser Server...")
    print(f"---------------------------------------------------------")
    
    server = viser.ViserServer(port=args.port, share=not args.no_share)
    
    # Dictionary to map filename -> full path
    # We prepend the dataset name to avoid confusion if multiple datasets have "raw.ply"
    file_options = {}
    for f in ply_files:
        # Format: "DatasetName - FileName"
        # e.g. "dishes - top95percent.ply"
        dataset_name = f.parent.parent.parent.name 
        key = f"{dataset_name} | {f.name}"
        file_options[key] = f

    file_names = sorted(list(file_options.keys()))

    # State Management
    current_handle = None
    # We store the current object center to orient new clients correctly
    current_center = np.array([0.0, 0.0, 0.0])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # When you connect, look at the current object center
        client.camera.look_at = current_center

    def load_ply(name):
        nonlocal current_handle, current_center
        path = file_options[name]
        
        print(f"Loading: {name}...")
        try:
            mesh = trimesh.load(path)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            return

        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                return

        points = mesh.vertices
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3]
        else:
            colors = np.ones_like(points) * 255

        if current_handle is not None:
            current_handle.remove()

        current_handle = server.scene.add_point_cloud(
            name="point_cloud",
            points=points,
            colors=colors,
            point_size=0.015,
        )
        
        # Reset camera for all connected clients (The Fix)
        if len(points) > 0:
            center = np.mean(points, axis=0)
            current_center = center # Update global state for new clients
            for client in server.get_clients().values():
                client.camera.look_at = center

    # GUI Elements
    with server.gui.add_folder("Controls"):
        gui_file = server.gui.add_dropdown(
            "Select File",
            options=file_names,
            # Default to the last one (usually the best filtered one)
            initial_value=file_names[-1] 
        )
        gui_size = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.1,
            step=0.001,
            initial_value=0.015
        )

    @gui_file.on_update
    def _(_):
        load_ply(gui_file.value)

    @gui_size.on_update
    def _(_):
        if current_handle is not None:
            current_handle.point_size = gui_size.value

    # Initial Load
    load_ply(file_names[-1])

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()