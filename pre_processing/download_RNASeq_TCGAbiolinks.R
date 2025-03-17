# Download FPKM-UQ value from GDC data portal
library(TCGAbiolinks)
library(SummarizedExperiment)

master.dir <- "."
save.dir <- file.path(master.dir, "gene_expression_FPKM_UQ")

cancer.types <- c("LUAD", "LUSC", "BRCA", "GBM", "COAD", "KIRC", "PAAD", "PRAD")

gene.numbers <- c()
protein.numbers <- c()
patient.numbers <- c()

for (cancer in cancer.types){
  data <- fpkm.data <- NULL
  query <- GDCquery(
    project = paste0("TCGA-", cancer),
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "STAR - Counts"
  )

  GDCdownload(query = query)
  data <- GDCprepare(query = query)
  data <- data[which(rowData(data)$gene_type %in% c("protein_coding", 'miRNA', "lncRNA")), ]
  fpkm.data <- assays(data)$fpkm_uq_unstrand
  rownames(fpkm.data) <- rowData(data)$gene_name
  fpkm.data <- fpkm.data[which(rowMedians(fpkm.data)>0), ]
  gene.types <- rowData(data)$gene_type[match(rownames(fpkm.data), rowData(data)$gene_name)]
  gene.numbers <- c(gene.numbers, nrow(fpkm.data))
  protein.numbers <- c(protein.numbers, table(gene.types)['protein_coding'])
  patient.numbers <- c(patient.numbers, ncol(fpkm.data))

  write.table(fpkm.data, paste0(save.dir, "/", cancer, ".txt"), sep = " ")
}

df_gene_number <- data.frame(cancer = cancer.types, n_gene = gene.numbers, n_protein_coding = protein.numbers, n_patient = patient.numbers)

write.csv(df_gene_number, paste0(save.dir, "/", "gene_number_summary_3.csv"))

