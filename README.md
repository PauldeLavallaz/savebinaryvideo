# 💾 Save Binary Video Node

Custom node para ComfyUI que guarda datos binarios (por ejemplo, la respuesta de un endpoint de video) en un archivo MP4.

## Instalación
1. Copiar la carpeta `SaveBinaryVideoNode` dentro de `ComfyUI/custom_nodes/`.
2. Reiniciar ComfyUI.
3. Buscar el nodo **💾 Save Binary Video** en la categoría `Morfeo/IO`.

## Uso
Conectar la salida binaria (por ejemplo, del nodo `Get Request Node`) a `binary_data`.
Configurar `filename` y `folder` a gusto.
El nodo guardará el archivo en el directorio especificado.
