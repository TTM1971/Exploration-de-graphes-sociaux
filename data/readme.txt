Fichiers :

nodeId.edges : Les arêtes du réseau égocentré du nœud « nodeId ». Les arêtes sont non orientées pour Facebook et orientées (A suit B) pour Twitter et Google+. Le nœud « ego » n’apparaît pas, mais on suppose qu’il suit chaque identifiant de nœud présent dans ce fichier.

nodeId.circles : Ensemble des cercles associés au nœud principal. Chaque ligne représente un cercle, constitué d'une série d'identifiants de nœuds. Le premier élément de chaque ligne correspond au nom du cercle.

nodeId.feat : Les caractéristiques de chacun des nœuds qui apparaissent dans le fichier d'arêtes.

nodeId.egofeat : Les fonctionnalités pour l'utilisateur ego.

nodeId.featnames : Noms des différentes dimensions de caractéristiques. Les caractéristiques sont égales à « 1 » si l’utilisateur possède cette propriété dans son profil, et à « 0 » sinon. Ce fichier a été anonymisé pour les utilisateurs Facebook, car les noms des caractéristiques révéleraient des données privées.