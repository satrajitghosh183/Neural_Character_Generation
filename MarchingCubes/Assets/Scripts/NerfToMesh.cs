using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class NerfToMesh : MonoBehaviour
{
    [Header("NERF Input Settings")]
    [SerializeField] private string nerfDensityFilePath = "Assets/NerfOutput/density.raw";
    [SerializeField] private string nerfColorFilePath = "Assets/NerfOutput/color.raw";
    [SerializeField] private Vector3Int volumeResolution = new Vector3Int(128, 128, 128);
    [SerializeField] private float densityThreshold = 0.5f;

    [Header("Mesh Generation Settings")]
    [SerializeField] private float meshScale = 1.0f;
    [SerializeField] private bool smoothNormals = true;
    [SerializeField] private bool generateUVs = true;
    [SerializeField] private bool createTexture = true;
    [SerializeField] private int maxVertices = 65000; // For mesh limitations
    
    [Header("Debug")]
    [SerializeField] private bool visualizeDensityField = false;
    [SerializeField] private float debugPointSize = 0.05f;
    [SerializeField] private int debugMaxPoints = 1000;

    // Internal data
    private float[,,] densityField;
    private Color[,,] colorField;
    private List<Vector3> vertices = new List<Vector3>();
    private List<int> triangles = new List<int>();
    private List<Vector3> normals = new List<Vector3>();
    private List<Vector2> uvs = new List<Vector2>();
    private List<Color> vertexColors = new List<Color>();
    
    private MeshFilter meshFilter;
    private MeshRenderer meshRenderer;

    void Start()
    {
        meshFilter = GetComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>();
        
        LoadNerfData();
        GenerateMesh();
        
        if (createTexture)
            CreateTextureFromColors();
    }

    private void LoadNerfData()
    {
        // In a real implementation, you would parse the actual NERF output format
        // This is a placeholder that would be replaced with actual NERF data loading
        Debug.Log("Loading NERF data...");
        
        // Initialize density and color fields
        densityField = new float[volumeResolution.x, volumeResolution.y, volumeResolution.z];
        colorField = new Color[volumeResolution.x, volumeResolution.y, volumeResolution.z];
        
        // For testing: Generate a simple character-like shape
        // This would be replaced by loading actual NERF data
        GenerateTestCharacter();
    }

    private void GenerateTestCharacter()
    {
        // Create a simple humanoid shape for testing
        Vector3 center = new Vector3(volumeResolution.x/2, volumeResolution.y/2, volumeResolution.z/2);
        float headRadius = volumeResolution.x * 0.12f;
        float bodyWidth = volumeResolution.x * 0.2f;
        float bodyHeight = volumeResolution.y * 0.3f;
        float bodyDepth = volumeResolution.z * 0.15f;
        float limbWidth = volumeResolution.x * 0.07f;
        
        // Fill density field with a humanoid shape
        for (int x = 0; x < volumeResolution.x; x++)
        {
            for (int y = 0; y < volumeResolution.y; y++)
            {
                for (int z = 0; z < volumeResolution.z; z++)
                {
                    Vector3 pos = new Vector3(x, y, z);
                    
                    // Head (sphere)
                    Vector3 headCenter = center + new Vector3(0, bodyHeight * 0.7f, 0);
                    float headDist = Vector3.Distance(pos, headCenter);
                    
                    // Body (box)
                    bool inBody = Mathf.Abs(x - center.x) < bodyWidth/2 && 
                                  y > center.y - bodyHeight/4 && 
                                  y < center.y + bodyHeight/2 &&
                                  Mathf.Abs(z - center.z) < bodyDepth/2;
                    
                    // Arms
                    bool inLeftArm = Mathf.Abs(x - (center.x - bodyWidth/2 - limbWidth/2)) < limbWidth/2 &&
                                     y > center.y && 
                                     y < center.y + bodyHeight/2 &&
                                     Mathf.Abs(z - center.z) < limbWidth/2;
                                     
                    bool inRightArm = Mathf.Abs(x - (center.x + bodyWidth/2 + limbWidth/2)) < limbWidth/2 &&
                                      y > center.y && 
                                      y < center.y + bodyHeight/2 &&
                                      Mathf.Abs(z - center.z) < limbWidth/2;
                    
                    // Legs
                    bool inLeftLeg = Mathf.Abs(x - (center.x - bodyWidth/4)) < limbWidth/2 &&
                                     y > center.y - bodyHeight/4 - bodyHeight/2 && 
                                     y < center.y - bodyHeight/4 &&
                                     Mathf.Abs(z - center.z) < limbWidth/2;
                                     
                    bool inRightLeg = Mathf.Abs(x - (center.x + bodyWidth/4)) < limbWidth/2 &&
                                      y > center.y - bodyHeight/4 - bodyHeight/2 && 
                                      y < center.y - bodyHeight/4 &&
                                      Mathf.Abs(z - center.z) < limbWidth/2;
                    
                    // Set density
                    if (headDist < headRadius || inBody || inLeftArm || inRightArm || inLeftLeg || inRightLeg)
                    {
                        densityField[x, y, z] = 1.0f;
                        
                        // Add some random coloration for testing
                        if (headDist < headRadius) // Head
                            colorField[x, y, z] = new Color(0.9f, 0.75f, 0.65f);
                        else if (inBody) // Body
                            colorField[x, y, z] = new Color(0.2f, 0.4f, 0.8f);
                        else // Arms and legs
                            colorField[x, y, z] = new Color(0.2f, 0.2f, 0.7f);
                    }
                    else
                    {
                        densityField[x, y, z] = 0.0f;
                        colorField[x, y, z] = Color.black;
                    }
                    
                    // Add some noise to make it look more natural
                    densityField[x, y, z] += (Random.value - 0.5f) * 0.1f;
                }
            }
        }
        
        Debug.Log("Test character density field generated");
    }

    private void GenerateMesh()
    {
        Debug.Log("Generating mesh from density field...");
        vertices.Clear();
        triangles.Clear();
        normals.Clear();
        uvs.Clear();
        vertexColors.Clear();
        
        // Marching cubes implementation
        for (int x = 0; x < volumeResolution.x - 1; x++)
        {
            for (int y = 0; y < volumeResolution.y - 1; y++)
            {
                for (int z = 0; z < volumeResolution.z - 1; z++)
                {
                    // Check for mesh size limits
                    if (vertices.Count > maxVertices - 12) // 12 possible new verts per cube
                    {
                        Debug.LogWarning("Reached maximum vertex count! Stopping mesh generation.");
                        break;
                    }
                    
                    MarchCube(new Vector3Int(x, y, z));
                }
            }
        }
        
        // Create the mesh
        Mesh mesh = new Mesh();
        
        // May need multiple submeshes if vertex count is high
        if (vertices.Count > 65535)
        {
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        }
        
        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        
        if (smoothNormals)
        {
            mesh.RecalculateNormals();
        }
        else if (normals.Count == vertices.Count)
        {
            mesh.normals = normals.ToArray();
        }
        
        if (generateUVs && uvs.Count == vertices.Count)
        {
            mesh.uv = uvs.ToArray();
        }
        
        mesh.colors = vertexColors.ToArray();
        
        mesh.RecalculateBounds();
        
        // Set mesh and scale
        meshFilter.mesh = mesh;
        transform.localScale = new Vector3(meshScale, meshScale, meshScale);
        
        Debug.Log($"Mesh generated with {vertices.Count} vertices and {triangles.Count/3} triangles");
    }

    private void MarchCube(Vector3Int position)
    {
        // Get the density values at the cube's corners
        float[] cubeCorners = new float[8];
        Color[] cornerColors = new Color[8];
        
        for (int i = 0; i < 8; i++)
        {
            Vector3Int corner = position + MarchingTable.Corners[i];
            
            // Ensure we're within bounds
            if (corner.x < volumeResolution.x && corner.y < volumeResolution.y && corner.z < volumeResolution.z)
            {
                cubeCorners[i] = densityField[corner.x, corner.y, corner.z];
                cornerColors[i] = colorField[corner.x, corner.y, corner.z];
            }
        }
        
        // Get the configuration index based on corner values
        int configIndex = GetConfigIndex(cubeCorners);
        
        // Skip empty configurations or full configurations
        if (configIndex == 0 || configIndex == 255)
        {
            return;
        }
        
        int edgeIndex = 0;
        
        // Process triangles for this configuration
        for (int t = 0; t < 5; t++) // Up to 5 triangles per cube
        {
            for (int v = 0; v < 3; v++) // 3 vertices per triangle
            {
                int triTableValue = MarchingTable.Triangles[configIndex, edgeIndex];
                
                if (triTableValue == -1)
                {
                    return;
                }
                
                // Get the edge start and end
                Vector3 edgeStart = position + MarchingTable.Edges[triTableValue, 0];
                Vector3 edgeEnd = position + MarchingTable.Edges[triTableValue, 1];
                
                // Determine which cube corners this edge connects
                int startCornerIdx = GetCornerFromEdge(triTableValue, 0);
                int endCornerIdx = GetCornerFromEdge(triTableValue, 1);
                
                // Interpolate vertex position based on density values
                float startVal = cubeCorners[startCornerIdx];
                float endVal = cubeCorners[endCornerIdx];
                float t_val = (densityThreshold - startVal) / (endVal - startVal);
                
                // Clamp to avoid division by zero
                if (float.IsNaN(t_val))
                    t_val = 0.5f;
                    
                t_val = Mathf.Clamp01(t_val);
                
                // Calculate interpolated vertex position
                Vector3 vertex = Vector3.Lerp(edgeStart, edgeEnd, t_val);
                
                // Normalize to 0-1 range for UV mapping
                Vector2 uv = new Vector2(
                    vertex.x / volumeResolution.x,
                    vertex.y / volumeResolution.y
                );
                
                // Interpolate color from corner colors
                Color vertexColor = Color.Lerp(cornerColors[startCornerIdx], cornerColors[endCornerIdx], t_val);
                
                // Add vertex data
                vertices.Add(vertex);
                triangles.Add(vertices.Count - 1);
                uvs.Add(uv);
                vertexColors.Add(vertexColor);
                
                edgeIndex++;
            }
        }
    }

    private int GetConfigIndex(float[] cubeCorners)
    {
        int configIndex = 0;
        
        for (int i = 0; i < 8; i++)
        {
            if (cubeCorners[i] > densityThreshold)
            {
                configIndex |= 1 << i;
            }
        }
        
        return configIndex;
    }

    private int GetCornerFromEdge(int edgeIndex, int startOrEnd)
    {
        // This maps edge indices to the corners they connect
        int[,] edgeToCornerMap = new int[12, 2] {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Bottom face
            {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top face
            {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Connecting edges
        };
        
        return edgeToCornerMap[edgeIndex, startOrEnd];
    }

    private void CreateTextureFromColors()
    {
        // Create a texture based on vertex colors - simple UV mapping
        int texWidth = 512;
        int texHeight = 512;
        Texture2D texture = new Texture2D(texWidth, texHeight);
        Color[] texPixels = new Color[texWidth * texHeight];
        
        // Simple mapping strategy - using UVs
        for (int i = 0; i < vertices.Count; i++)
        {
            Vector2 uv = uvs[i];
            int xCoord = Mathf.Clamp(Mathf.FloorToInt(uv.x * texWidth), 0, texWidth-1);
            int yCoord = Mathf.Clamp(Mathf.FloorToInt(uv.y * texHeight), 0, texHeight-1);
            int pixelIndex = yCoord * texWidth + xCoord;
            
            if (pixelIndex >= 0 && pixelIndex < texPixels.Length)
            {
                texPixels[pixelIndex] = vertexColors[i];
            }
        }
        
        // Fill in missing pixels with neighbors' average
        FillTextureGaps(texPixels, texWidth, texHeight);
        
        texture.SetPixels(texPixels);
        texture.Apply();
        
        // Create material with texture
        Material material = new Material(Shader.Find("Standard"));
        material.mainTexture = texture;
        meshRenderer.material = material;
        
        Debug.Log("Character texture created");
    }

    private void FillTextureGaps(Color[] pixels, int width, int height)
    {
        Color[] filledPixels = new Color[pixels.Length];
        System.Array.Copy(pixels, filledPixels, pixels.Length);
        
        // Simple gap filling - average of non-black neighbors
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                if (pixels[idx] == Color.black)
                {
                    List<Color> neighbors = new List<Color>();
                    
                    // Check 8 neighbors
                    for (int ny = Mathf.Max(0, y-1); ny <= Mathf.Min(height-1, y+1); ny++)
                    {
                        for (int nx = Mathf.Max(0, x-1); nx <= Mathf.Min(width-1, x+1); nx++)
                        {
                            int nidx = ny * width + nx;
                            if (pixels[nidx] != Color.black)
                            {
                                neighbors.Add(pixels[nidx]);
                            }
                        }
                    }
                    
                    // Compute average color of non-black neighbors
                    if (neighbors.Count > 0)
                    {
                        Color avg = Color.black;
                        foreach (Color c in neighbors)
                        {
                            avg += c;
                        }
                        avg /= neighbors.Count;
                        filledPixels[idx] = avg;
                    }
                }
            }
        }
        
        System.Array.Copy(filledPixels, pixels, pixels.Length);
    }

    void OnDrawGizmosSelected()
    {
        if (!visualizeDensityField || densityField == null || !Application.isPlaying)
            return;
            
        Gizmos.color = Color.green;
        int pointCount = 0;
        
        // Skip some voxels to avoid too many gizmos
        int skip = Mathf.Max(1, volumeResolution.x / 20);
        
        for (int x = 0; x < volumeResolution.x; x += skip)
        {
            for (int y = 0; y < volumeResolution.y; y += skip)
            {
                for (int z = 0; z < volumeResolution.z; z += skip)
                {
                    if (densityField[x, y, z] > densityThreshold)
                    {
                        Gizmos.color = colorField[x, y, z];
                        Gizmos.DrawSphere(
                            transform.position + new Vector3(x, y, z) * meshScale, 
                            debugPointSize * meshScale
                        );
                        
                        if (++pointCount >= debugMaxPoints)
                            return;
                    }
                }
            }
        }
    }

    // Method to directly use NERF output
    public void ImportNerfOutput(float[,,] densities, Color[,,] colors)
    {
        // Check dimensions
        if (densities.GetLength(0) != volumeResolution.x ||
            densities.GetLength(1) != volumeResolution.y ||
            densities.GetLength(2) != volumeResolution.z)
        {
            Debug.LogError("Imported NERF data dimensions do not match the configured resolution!");
            return;
        }
        
        // Copy data
        for (int x = 0; x < volumeResolution.x; x++)
        {
            for (int y = 0; y < volumeResolution.y; y++)
            {
                for (int z = 0; z < volumeResolution.z; z++)
                {
                    densityField[x, y, z] = densities[x, y, z];
                    colorField[x, y, z] = colors[x, y, z];
                }
            }
        }
        
        // Generate mesh with new data
        GenerateMesh();
        
        if (createTexture)
            CreateTextureFromColors();
    }
}