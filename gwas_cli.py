
from gwas_class_auto import *
import sys


dictionary = {k.replace('-', ''):v for k,v in [x.split('=') for x in sys.argv] }


df = pd.read_csv(dictionary['df']).drop_duplicates(subset = 'rfid')
gwas = gwas_pipe(path = f'{dictionary["name"]}/',
             all_genotypes = dictionary['genotypes'], #'round9_1.vcf.gz',
             data = dictionary['df'],
             project_name = dictionary['name'],
             traits = dictionary['traits'].split('__'),
             threads= dictionary['threads'])
gwas.SubsampleMissMafHweFilter(**dictionary)
gwas.generateGRM(**dictionary)
gwas.snpHeritability(**dictionary)
gwas.GWAS(**dictionary)
gwas.addGWASresultsToDb(**dictionary)
qtls = gwas.callQTLs(**dictionary)
gwas.annotate(qtls, **dictionary)
gwas.eQTL(qtls, **dictionary)
gwas.print_watermark()
