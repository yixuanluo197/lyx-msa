# 第09组执行交接

本批路径：/data2/yb/paper/lyx-msa/CMERC/experiments/batch_size_diagnosis/20261024T131956Z

Supervisor PID2032912；CPU finalizer PID2032913。唯一正式run MELD_cmerc_batch8_lr1e-4_seed2024。当前GPU准入排队，不得复制启动器或重复提交。state/status.json记录唯一计数及完整PID/资源。冻结清单39项SHA，hash见provenance/supervisor_launch.json。04–08共10根、1691文件/30symlink保存于history_before。

通过门禁后自动独立首Adam探针→GPU释放→重新3次准入→唯一正式40epoch；正式从头初始化不复用探针权重。当前新外部PID存在，门禁暂停；不改变科学配置、不终止任务、不迁移服务器。

完成后CPU finalizer自动执行verify_lr.py（无forward）、analyze_batch_size.py、plot_batch_size.py；state/finalization.json标记真实状态，脚本失败仅排错CPU处理，不重训。报告路径reports/CMERC_BATCH_SIZE_DIAGNOSIS.md及仓库根同名；最终必须检查两个Checkpoint/40train40val/0test/历史hash/图和字段。

原--batch-size同时改变train/validation分组，报告必须披露未pack双向GRU/padding可能影响验证预测。主判据best↑且末10不降为下一Seed候选；只建议，不自动启动额外实验或测试。当前用户未请求09上传GitHub。

CPU delivery watcher PID2063824：仅在finalization=completed且独立验收通过后更新09档案导航。资源准入状态报告已保存于仓库根reports/CMERC_BATCH_SIZE_DIAGNOSIS.md；不得称为完整实验结果。
