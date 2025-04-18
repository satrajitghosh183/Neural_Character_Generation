using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

public class NerfOutputParser : MonoBehaviour
{
    [Header("NERF Input")]
    [SerializeField] private string nerfOutputFolder = "Assets/NerfOutput/";
    [SerializeField] private string densityFilename = "density.raw";
    [SerializeField] private string colorFilename = "rgb.raw";

    [Header("NERF Parameters")]
    [SerializeField] private Vector3Int volumeResolution = new Vector3Int(128, 128, 128);
    [SerializeField] private bool normalizeValues = true;
    [SerializeField] private bool invertDensity = false;

    [Header("Output")]
    [SerializeField] private NerfToMesh meshGenerator;

    public enum NerfFormat { RawBinary, CSV, JSON, NPY }
    [SerializeField] private NerfFormat nerfFormat = NerfFormat.RawBinary;

    void Start()
    {
        if (meshGenerator == null)
        {
            meshGenerator = GetComponent<NerfToMesh>();
            if (meshGenerator == null)
            {
                Debug.LogError("Missing NerfToMesh component.");
                return;
            }
        }

        ParseNerfOutput();
    }

    public void ParseNerfOutput()
    {
        try
        {
            float[,,] densities = new float[volumeResolution.x, volumeResolution.y, volumeResolution.z];
            Color[,,] colors = new Color[volumeResolution.x, volumeResolution.y, volumeResolution.z];

            switch (nerfFormat)
            {
                case NerfFormat.RawBinary:
                    ParseRawBinary(ref densities, ref colors);
                    break;
                case NerfFormat.CSV:
                    ParseCSV(ref densities, ref colors);
                    break;
                case NerfFormat.JSON:
                case NerfFormat.NPY:
                    Debug.LogWarning($"{nerfFormat} format not implemented.");
                    GeneratePlaceholderData(ref densities, ref colors);
                    break;
            }

            if (normalizeValues) NormalizeData(ref densities);
            if (invertDensity) InvertDensity(ref densities);

            meshGenerator.ImportNerfOutput(densities, colors);
            Debug.Log("Parsing complete.");
        }
        catch (Exception e)
        {
            Debug.LogError($"Parsing failed: {e.Message}\n{e.StackTrace}");
        }
    }

    private void ParseRawBinary(ref float[,,] densities, ref Color[,,] colors)
    {
        string densityPath = Path.Combine(nerfOutputFolder, densityFilename);
        string colorPath = Path.Combine(nerfOutputFolder, colorFilename);

        if (!File.Exists(densityPath))
        {
            Debug.LogWarning("Missing density.raw, using placeholder.");
            GeneratePlaceholderData(ref densities, ref colors);
            return;
        }

        using (BinaryReader reader = new BinaryReader(File.Open(densityPath, FileMode.Open)))
        {
            for (int z = 0; z < volumeResolution.z; z++)
            for (int y = 0; y < volumeResolution.y; y++)
            for (int x = 0; x < volumeResolution.x; x++)
                densities[x, y, z] = reader.BaseStream.Position + 4 <= reader.BaseStream.Length ? reader.ReadSingle() : 0f;
        }

        if (File.Exists(colorPath))
        {
            using (BinaryReader reader = new BinaryReader(File.Open(colorPath, FileMode.Open)))
            {
                for (int z = 0; z < volumeResolution.z; z++)
                for (int y = 0; y < volumeResolution.y; y++)
                for (int x = 0; x < volumeResolution.x; x++)
                {
                    if (reader.BaseStream.Position + 12 <= reader.BaseStream.Length)
                    {
                        float r = reader.ReadSingle();
                        float g = reader.ReadSingle();
                        float b = reader.ReadSingle();
                        colors[x, y, z] = new Color(r, g, b, 1.0f);
                    }
                    else
                    {
                        colors[x, y, z] = Color.gray;
                    }
                }
            }
        }
        else
        {
            Debug.LogWarning("Missing rgb.raw, applying default colors.");
            GenerateDefaultColors(ref densities, ref colors);
        }
    }

    private void ParseCSV(ref float[,,] densities, ref Color[,,] colors)
    {
        string densityPath = Path.Combine(nerfOutputFolder, densityFilename.Replace(".raw", ".csv"));
        string colorPath = Path.Combine(nerfOutputFolder, colorFilename.Replace(".raw", ".csv"));

        if (!File.Exists(densityPath))
        {
            Debug.LogWarning("Missing density.csv, using placeholder.");
            GeneratePlaceholderData(ref densities, ref colors);
            return;
        }

        string[] lines = File.ReadAllLines(densityPath);
        int index = 0;

        for (int z = 0; z < volumeResolution.z; z++)
        for (int y = 0; y < volumeResolution.y; y++)
        for (int x = 0; x < volumeResolution.x; x++)
            densities[x, y, z] = (index < lines.Length && float.TryParse(lines[index++], out float val)) ? val : 0f;

        if (File.Exists(colorPath))
        {
            lines = File.ReadAllLines(colorPath);
            index = 0;

            for (int z = 0; z < volumeResolution.z; z++)
            for (int y = 0; y < volumeResolution.y; y++)
            for (int x = 0; x < volumeResolution.x; x++)
            {
                if (index < lines.Length)
                {
                    string[] rgb = lines[index++].Split(',');
                    if (rgb.Length >= 3 &&
                        float.TryParse(rgb[0], out float r) &&
                        float.TryParse(rgb[1], out float g) &&
                        float.TryParse(rgb[2], out float b))
                        colors[x, y, z] = new Color(r, g, b, 1.0f);
                    else
                        colors[x, y, z] = Color.gray;
                }
            }
        }
        else
        {
            Debug.LogWarning("Missing color.csv, applying default colors.");
            GenerateDefaultColors(ref densities, ref colors);
        }
    }

    private void GeneratePlaceholderData(ref float[,,] densities, ref Color[,,] colors)
    {
        Vector3 center = new Vector3(volumeResolution.x, volumeResolution.y, volumeResolution.z) / 2f;
        float radius = volumeResolution.x * 0.25f;

        for (int z = 0; z < volumeResolution.z; z++)
        for (int y = 0; y < volumeResolution.y; y++)
        for (int x = 0; x < volumeResolution.x; x++)
        {
            float dist = Vector3.Distance(new Vector3(x, y, z), center);
            densities[x, y, z] = dist < radius ? 1.0f : 0.0f;
            colors[x, y, z] = Color.Lerp(Color.red, Color.blue, dist / radius);
        }
    }

    private void GenerateDefaultColors(ref float[,,] densities, ref Color[,,] colors)
    {
        for (int z = 0; z < volumeResolution.z; z++)
        for (int y = 0; y < volumeResolution.y; y++)
        for (int x = 0; x < volumeResolution.x; x++)
        {
            float val = densities[x, y, z];
            colors[x, y, z] = new Color(val, val, val, 1.0f);
        }
    }

    private void NormalizeData(ref float[,,] densities)
    {
        float min = float.MaxValue;
        float max = float.MinValue;

        foreach (float d in densities)
        {
            if (d < min) min = d;
            if (d > max) max = d;
        }

        float range = Mathf.Max(max - min, 1e-5f);
        for (int z = 0; z < volumeResolution.z; z++)
        for (int y = 0; y < volumeResolution.y; y++)
        for (int x = 0; x < volumeResolution.x; x++)
            densities[x, y, z] = (densities[x, y, z] - min) / range;
    }

    private void InvertDensity(ref float[,,] densities)
    {
        for (int z = 0; z < volumeResolution.z; z++)
        for (int y = 0; y < volumeResolution.y; y++)
        for (int x = 0; x < volumeResolution.x; x++)
            densities[x, y, z] = 1.0f - densities[x, y, z];
    }
}
