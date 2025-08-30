import xml.etree.ElementTree as ET

def modifier_masse_inertie(urdf_path, link_name, nouvelle_masse, nouvelle_inertie):
    """
    Modifie la masse et les inerties d'un link dans un fichier URDF.

    urdf_path : chemin vers le fichier URDF
    link_name : nom du link à modifier
    nouvelle_masse : masse (float)
    nouvelle_inertie : dictionnaire avec ixx, iyy, izz, ixy, ixz, iyz
    """
    # Charger l'arbre XML
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Chercher le link
    for link in root.findall('link'):
        if link.attrib.get('name') == link_name:
            inertial = link.find('inertial')
            if inertial is None:
                # Créer le bloc inertial si absent
                inertial = ET.SubElement(link, 'inertial')

            # Modifier ou créer <mass>
            mass_elem = inertial.find('mass')
            if mass_elem is None:
                mass_elem = ET.SubElement(inertial, 'mass')
            mass_elem.set('value', str(nouvelle_masse))

            # Modifier ou créer <inertia>
            inertia_elem = inertial.find('inertia')
            if inertia_elem is None:
                inertia_elem = ET.SubElement(inertial, 'inertia')

            for key, val in nouvelle_inertie.items():
                inertia_elem.set(key, str(val))

    # Sauvegarder dans le même fichier
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"Masse et inerties modifiées pour le link '{link_name}'.")

# Exemple d'utilisation :
modifier_masse_inertie(
    urdf_path="robot.urdf",
    link_name="drone_body",
    nouvelle_masse=5.0,
    nouvelle_inertie={
        "ixx": 0.05, "iyy": 0.05, "izz": 0.05,
        "ixy": 0.0, "ixz": 0.0, "iyz": 0.0
    }
)

