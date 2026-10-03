"""Export independently authored HiVE hierarchies with explicit parent links."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'text_prompts/hive_crc_blca_20261003_v2'

# Each entry is a coarse morphological concept and exactly three subordinate
# concepts. This is authored source text, not a shared-clause template corpus.
BANKS = {
 'crc': {
  'ADENO_NOS': [
   ('Irregular Malignant Glands', 'Variably sized, distorted glands infiltrate colorectal tissue, with crowded epithelial structures and disrupted normal mucosal organization.', [
    ('Distorted Lumina', 'Uneven glandular lumina are surrounded by atypical columnar epithelium with irregular epithelial contours.'),
    ('Nuclear Stratification', 'Elongated hyperchromatic nuclei overlap and stratify along the lining of malignant glands.'),
    ('Loss of Polarity', 'Tumor cells show disordered nuclear orientation and variable positioning relative to the glandular lumen.')]),
   ('Cribriform Tumor Architecture', 'Confluent malignant glands form complex epithelial structures containing multiple closely spaced luminal openings.', [
    ('Epithelial Bridges', 'Atypical epithelial bridges divide confluent glandular spaces into multiple smaller luminal compartments.'),
    ('Back-to-Back Glands', 'Closely packed malignant glands have little intervening stroma and irregular shared epithelial boundaries.'),
    ('Crowded Atypical Epithelium', 'Dense epithelial proliferations surrounding cribriform openings show overlapping hyperchromatic nuclei and variable nuclear size.')]),
   ('Desmoplastic Invasive Front', 'Angulated tumor glands and small epithelial groups infiltrate a reactive fibrous stromal compartment.', [
    ('Angulated Infiltrating Glands', 'Irregular, sharply angulated malignant glands extend between collagen bundles at the tumor–stroma interface.'),
    ('Small Epithelial Groups', 'Small clusters of atypical epithelial cells lie within reactive stroma near larger infiltrating glands.'),
    ('Reactive Fibroblastic Stroma', 'Collagen-rich tissue containing spindle-shaped fibroblasts surrounds invasive malignant glandular structures.')]),
   ('Necrotic Glandular Lumina', 'Malignant glandular structures contain granular necrotic material, producing irregular intraluminal debris.', [
    ('Granular Cellular Debris', 'Glandular lumina contain granular eosinophilic material mixed with degenerating epithelial fragments.'),
    ('Karyorrhectic Fragments', 'Dark fragmented nuclear material is distributed within necrotic debris inside malignant glands.'),
    ('Viable Tumor Lining', 'Atypical viable epithelial cells border luminal necrotic material, preserving recognizable malignant gland outlines.')])],
  'MUCINOUS': [
   ('Extracellular Mucin Lakes', 'Broad pools of extracellular mucin separate malignant epithelial elements and replace portions of the colorectal tissue architecture.', [
    ('Pale Mucinous Matrix', 'Abundant pale extracellular material occupies spaces between malignant epithelial clusters within a mucin lake.'),
    ('Sparse Tumor Distribution', 'Small viable tumor groups are separated by broad expanses of extracellular mucin.'),
    ('Mucin-Lake Boundaries', 'The margins of extracellular mucin collections abut residual tissue, tumor epithelium, or fibrous stromal partitions.')]),
   ('Floating Malignant Clusters', 'Detached epithelial islands and irregular tumor aggregates are suspended within extracellular mucin collections.', [
    ('Suspended Epithelial Islands', 'Compact aggregates of atypical epithelial cells have free surfaces surrounded by extracellular mucin.'),
    ('Atypical Cluster Nuclei', 'Floating tumor aggregates contain enlarged, irregular, hyperchromatic nuclei with variable crowding.'),
    ('Isolated Mucin-Associated Cells', 'Individual atypical epithelial cells or tiny detached groups occur between larger clusters in extracellular mucin.')]),
   ('Mucin-Associated Gland Fragments', 'Incomplete glands and epithelial strips border or traverse mucinous regions, retaining variable glandular differentiation.', [
    ('Fragmented Gland Walls', 'Discontinuous segments of atypical glandular epithelium line portions of extracellular mucin spaces.'),
    ('Lumina in Epithelial Groups', 'Small glandular openings are visible within epithelial aggregates embedded in mucinous material.'),
    ('Mucin-Filled Tumor Cytoplasm', 'Some malignant epithelial cells contain pale cytoplasmic mucin that alters the position and shape of their nuclei.')]),
   ('Fibrous Septa Between Mucin Pools', 'Collagenous tissue partitions separate adjacent extracellular mucin collections and support variable tumor-associated epithelium.', [
    ('Collagenous Pool Partitions', 'Eosinophilic collagen bundles form narrow tissue boundaries between neighboring mucin lakes.'),
    ('Septal Malignant Epithelium', 'Atypical epithelial strips or small tumor groups lie along fibrous septa bordering mucinous spaces.'),
    ('Fibroblastic Septal Reaction', 'Spindle-shaped stromal cells and collagen are present in tissue partitions adjacent to mucin-associated malignant epithelium.')])]
 },
 'blca': {
  'NON_PAPILLARY': [
   ('Solid Urothelial Tumor Nests', 'Cohesive malignant urothelial groups form solid islands or broad sheets with irregular outlines in the surrounding tissue.', [
    ('Cohesive Atypical Epithelium', 'Closely apposed polygonal tumor cells form compact epithelial aggregates within a solid urothelial nest.'),
    ('Disordered Nest Organization', 'Tumor cells within solid nests have uneven orientation and irregular spacing rather than orderly epithelial layering.'),
    ('Uneven Nest Boundaries', 'The outer contours of malignant epithelial islands are irregular and may contain small projecting cell groups.')]),
   ('Infiltrative Cords and Small Groups', 'Narrow malignant epithelial cords and detached small tumor groups extend through stromal tissue at irregular interfaces.', [
    ('Thin Epithelial Cords', 'Short strands of atypical urothelial cells track between stromal collagen bundles.'),
    ('Detached Stromal Tumor Groups', 'Small malignant epithelial clusters are separated from larger tumor nests by intervening connective tissue.'),
    ('Angulated Infiltrative Profiles', 'Irregularly shaped epithelial groups form sharply contoured profiles where tumor interdigitates with stroma.')]),
   ('Reactive Stroma Around Tumor Islands', 'Fibrous stromal compartments surround malignant epithelial islands and separate neighboring solid tumor regions.', [
    ('Perinest Collagen Bundles', 'Eosinophilic collagen fibers border and partition cohesive malignant urothelial groups.'),
    ('Interface Fibroblasts', 'Spindle-shaped fibroblastic cells lie in connective tissue immediately adjacent to infiltrative tumor islands.'),
    ('Inflammation at Stromal Interfaces', 'Variable inflammatory cells are interspersed within the stroma around malignant epithelial nests.')]),
   ('Heterogeneous Solid Tumor Compartments', 'Solid tumor areas show variable cellular density and may contain focal necrotic regions beside viable malignant epithelium.', [
    ('Uneven Cellular Packing', 'Cell-dense epithelial areas alternate with less compact malignant groups within a solid tumor compartment.'),
    ('Focal Intratumoral Necrosis', 'Localized eosinophilic debris and fragmented nuclei occur within some solid malignant epithelial regions.'),
    ('Viable Perinecrotic Tumor', 'Atypical urothelial cells border focal necrotic material within an otherwise viable solid tumor area.')])],
  'PAPILLARY': [
   ('Branching Papillary Fronds', 'Exophytic epithelial projections form branching papillary structures supported by recognizable connective tissue cores.', [
    ('Fingerlike Epithelial Projections', 'Elongated epithelial fronds extend around stromal cores and project into spaces between adjacent papillae.'),
    ('Secondary Papillary Branches', 'Smaller epithelial-lined projections emerge from larger papillary stalks, creating a branched architecture.'),
    ('Interpapillary Spaces', 'Open spaces separate neighboring epithelial-lined fronds and delineate their external papillary contours.')]),
   ('Fibrovascular Papillary Cores', 'Connective tissue stalks containing small vessels support malignant urothelial layers within the papillary projections.', [
    ('Core Capillary Profiles', 'Small vascular channels containing erythrocytes are visible within the connective tissue center of papillary stalks.'),
    ('Collagenous Core Matrix', 'Fine collagen fibers and spindle-shaped stromal cells form the supporting tissue beneath papillary epithelium.'),
    ('Epithelial–Core Boundary', 'Malignant urothelial layers lie along a recognizable interface with the underlying fibrovascular stalk.')]),
   ('Atypical Urothelial Lining', 'Multilayered tumor epithelium covers papillary cores with variable thickness and disturbance of normal urothelial organization.', [
    ('Uneven Epithelial Thickness', 'The number and arrangement of urothelial cell layers vary along individual papillary fronds.'),
    ('Disordered Nuclear Alignment', 'Nuclei within the papillary lining show irregular orientation and variable crowding relative to the stromal core.'),
    ('Surface Maturation Changes', 'The epithelial surface shows variable loss of orderly maturation and irregular contours of superficial tumor cells.')]),
   ('Complex and Fused Papillary Structures', 'Adjacent papillary projections may become crowded, broadened, or fused while retaining portions of their supporting stromal framework.', [
    ('Crowded Papillary Profiles', 'Closely spaced epithelial-lined projections create complex contours with narrow intervening spaces.'),
    ('Epithelial Fusion Between Fronds', 'Tumor epithelium joins neighboring papillary projections, producing irregular bridges between their outlines.'),
    ('Residual Cores in Broad Projections', 'Small fibrovascular stromal profiles remain identifiable within broadened or partially fused papillary tumor structures.')])]
 }
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for cohort, classes in BANKS.items():
        bank, parents = {}, {}
        for label, concepts in classes.items():
            assert len(concepts) == 4
            branch = {name: text for name, text, _ in concepts}
            parents[label] = []
            for parent_index, (name, _, children) in enumerate(concepts):
                assert len(children) == 3
                for child, text in children:
                    branch[name + ' / ' + child] = text
                    parents[label].append({'parent_index': parent_index, 'parent': name, 'child': child})
            assert len(branch) == 16
            bank[label] = branch
        texts = [v for branch in bank.values() for v in branch.values()]
        assert len(set(texts)) == 32
        assert all(' may show ' not in t for t in texts)
        path = OUTPUT / cohort / 'hierarchy.json'
        path.parent.mkdir(exist_ok=True)
        content = (json.dumps(bank, indent=2, ensure_ascii=False) + '\n').encode()
        if path.exists() and path.read_bytes() != content:
            raise ValueError('Preserve versioned bank; use a new version for changes')
        path.write_bytes(content)
        records.append({'cohort': cohort, 'method': 'hive_mil', 'path': str(path.relative_to(ROOT)),
                        'classes': list(classes), 'sha256': hashlib.sha256(content).hexdigest(),
                        'parents': parents})
    provenance = {'version': 'hive_crc_blca_20261003_v2', 'provenance': 'locally_authored',
                  'authorship': 'Assistant-authored in this conversation; no external generation API called; exact model endpoint/revision not independently attested.',
                  'policy': 'Four morphological parent concepts and three explicitly authored children per parent. Possible H&E appearances, not necessary/sufficient diagnoses. No patient images, reports, outcomes, or test metrics used to select text.',
                  'upstream_structure_source': 'https://github.com/bryanwong17/HiVE-MIL',
                  'task_definition_source': '../crc_blca_gpt6_astra_v1/source.json',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'records': records}
    (OUTPUT / 'MANIFEST.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'banks': len(records), 'nodes': 64, 'output': str(OUTPUT)}))


if __name__ == '__main__':
    main()
